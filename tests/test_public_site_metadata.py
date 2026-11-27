from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from ouroboros.tools.release_sync import (
    RELEASE_ASSET_TEMPLATES,
    DESKTOP_DOWNLOAD_IDS,
    WINDOWS_SETUP_PROOF_ID,
    WINDOWS_ZIP_PROOF_ID,
    desktop_download_ids,
    download_links_include_windows_installer,
    is_development_marker,
    published_download_version,
    release_asset_download_url,
    resolve_release_windows_installer,
)

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
DOCS = REPO / "docs"
ORIGIN = "https://ouroboros-agent.ai"
NOVEMBER_SHA256 = "9976660610a53f0352a8df529943653de20d17f605741fe47a8ffc3fa8345709"
NOVEMBER_CARD_SHA256 = "15d6fbada68b212adcd9e483b99a29a1bf0811e104617a5f898030f21b2db779"
DECEMBER_SHA256 = "97f25397c30f6f3bf4ccbf9efdcef4c4495cee94d18f94c947e1141d4acaf253"
DECEMBER_CARD_SHA256 = "7f995bf3edb9ca3879ce5677d20ac452bb3d7ed6ee0bf459bc7ba92516670ca2"
# A published edition is frozen: its data and its card, and the words its archive page must carry in its head.
RATING_ARCHIVES = {
    "2026.11": ("November", NOVEMBER_CARD_SHA256, ("Edition 2026.11", "11 models", "32 kinds of work")),
    "2026.12": ("December", DECEMBER_CARD_SHA256, ("Edition 2026.12", "15 models", "6 columns")),
}
RATING_GRADES = {"S", "A", "B", "C"}
# Separate digits from word boundaries: a unit touching a number is still a unit (15руб., GBP25).
_RATING_UNITS = (
    r"usd|eur|rub|gbp|dollars?|euros?|cents?|roubles?|rubles?|рубл\w*|руб\.?|долл\.?|доллар\w*|евро|"
    r"цент(?:а|ов|у|ом|е|ы|ам|ами|ах)?|коп(?:\.|ей(?:ка|ки|ку|кой|кою|ками|ках|кам|ке)|еек)?|бакс\w*"
)
_RATING_AMOUNT = r"\d+(?:[.,]\d+)*(?:[\u00a0\u202f ]\d{3}(?:[.,]\d+)?)? *(?:[kmb](?![^\W\d_])|тыс\.?|млн|млрд)?"
RATING_MONEY = re.compile(
    rf"[$€₽£¥]|(?<![^\W\d_])(?:{_RATING_UNITS})\s*{_RATING_AMOUNT}"
    rf"|{_RATING_AMOUNT}\s*(?:{_RATING_UNITS})(?![^\W\d_])"
    # Preserve the existing policy against naming a currency or its sign even without a numeric amount.
    rf"|\b(?:{_RATING_UNITS})(?!\w)", re.I,
)
RATING_CARRIED_FIELDS = ("votes", "reviewers", "period", "volume", "totals.tokens")
RATING_COUNT = re.compile(r"\b([0-9]+)\s+(letters?|dashes?|cells? without a letter|букв[аы]?|прочерк(?:а|ов)?)\b", re.I)

FROZEN_RATING_JSON = {Path(f"data/editions/{name}/edition.json"): digest for name, digest in (
    ("2026.11", NOVEMBER_SHA256), ("2026.12", DECEMBER_SHA256),
)}
RATING_REVISION = re.compile(r"([0-9]{4}\.[0-9]{2})\.[1-9][0-9]*")
INDEXABLE = {
    "/": SITE / "index.html",
    "/about/": SITE / "about" / "index.html",
    "/privacy/": SITE / "privacy" / "index.html",
    "/terms/": SITE / "terms" / "index.html",
    "/paper/": SITE / "paper" / "index.html",
    "/community/": SITE / "community" / "index.html",
    "/install/": SITE / "install" / "index.html",
    "/benchmarks/": SITE / "benchmarks" / "index.html",
    "/history/first-48-hours/": SITE / "history" / "first-48-hours" / "index.html",
    "/llm-rating/": SITE / "public" / "llm-rating" / "index.html",
    "/llm-rating/editions.html": SITE / "public" / "llm-rating" / "editions.html",
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
        "Ouroboros-Setup-{version}-windows-x64.exe": ("windows", "x64"),
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
    windows_rows = [row for row in manifest["artifacts"] if row["platform"] == "windows"]
    assert [row["format"] for row in windows_rows] == ["exe", "zip"]
    assert windows_rows[0]["availability"] == "per-release"
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
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    source = (SITE / "install" / "index.html").read_text(encoding="utf-8")
    generated = (DOCS / "install" / "index.html").read_text(encoding="utf-8")
    assert source.split("<body", 1)[1] == generated.split("<body", 1)[1]
    published = published_download_version(source) if is_development_marker(version) else version
    has_setup = resolve_release_windows_installer(REPO, version, published_version=published)
    if has_setup is None:
        has_setup = download_links_include_windows_installer(readme, source, generated, version=published)
    assert has_setup is not None

    for home in (
        (SITE / "index.html").read_text(encoding="utf-8"),
        (DOCS / "index.html").read_text(encoding="utf-8"),
    ):
        assert '<a class="os-card" href="/install/#windows"><b>Windows</b><span>x64 · desktop</span></a>' in home
        assert "<b>Windows</b><span>x64 · .zip</span>" not in home

    for html in (source, generated):
        downloads = html.split('class="download-grid platform-downloads"', 1)[1].split(
            'class="download-proof-note"', 1
        )[0]
        assert '<article id="windows">' in downloads
        assert "/releases/latest" not in downloads
        windows_card = downloads.split('<article id="windows">', 1)[1].split("</article>", 1)[0]
        active_ids = desktop_download_ids(windows_installer=has_setup)
        for proof_id in active_ids:
            expected = release_asset_download_url(proof_id, published)
            assert f'data-release-download="{proof_id}"' in html
            assert expected in html
        if has_setup:
            assert downloads.index(f'data-release-download="{WINDOWS_SETUP_PROOF_ID}"') < downloads.index(
                f'data-release-download="{WINDOWS_ZIP_PROOF_ID}"'
            )
            assert "Run the signed per-user installer" in windows_card
        else:
            assert f'data-release-download="{WINDOWS_SETUP_PROOF_ID}"' not in html
            assert "Ouroboros-Setup-" not in html
            assert "Extract the ZIP to a permanent folder" in windows_card
            assert "installer" not in windows_card.lower() and "<code>.exe</code>" not in windows_card
        assert html.index("platform-downloads") < html.index(
            "Advanced: headless CLI with uv"
        ) < html.index("Develop or run from source")
        assert "verification evidence, not additional installers" in html
        assert "usr/lib/ouroboros/_internal/python-standalone/bin/python3" in html
        assert "remove the old ZIP folder" not in html
        assert "re-enable sign-in startup in Settings in the installed copy" in html
        assert "install-ouroboros-cli.cmd --force" in html


def test_generated_public_files_match_source():
    for name in ("CNAME", "robots.txt", "sitemap.xml", "llms.txt", "install.json"):
        assert (SITE / "public" / name).read_bytes() == (DOCS / name).read_bytes()


def test_readme_install_page_and_deployment_guide_name_the_factory_values():
    """What a newcomer reads before installing is the factory limits themselves, in one paragraph on every surface;
    an operator's deployment guide states the same factory values as rows of its first-run table."""
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS as defaults

    total, task, daily = (int(defaults[key]) for key in (
        "TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD", "OUROBOROS_CONSCIOUSNESS_DAILY_USD"))
    paragraph = (f"out of the box, new paid work stops once known spend reaches ${total} in total, or ${task} for one "
                 "task and its helpers; charges already in flight can pass a limit. Background consciousness starts with "
                 "the setup wizard's switch, which is on unless you turn it off (without the wizard it stays off until "
                 f"you start it), and then has ${daily} for each 24 hours. Settings changes all three.")
    guided = "The first-run wizard guides accounts and models, access, background consciousness, the skills to start with, and budget setup."
    for path in (REPO / "README.md", SITE / "install" / "index.html", DOCS / "install" / "index.html"):
        assert paragraph in path.read_text(encoding="utf-8"), path.relative_to(REPO)
        assert guided in path.read_text(encoding="utf-8"), path.relative_to(REPO)  # the wizard's steps, its Consciousness one
    table = (DOCS / "DEPLOYMENT.md").read_text(encoding="utf-8")
    assert f"| Limits | ${total} in total, ${task} for one task and its helpers, of known spend |" in table
    assert f"| Consciousness | off (${daily} over 24 hours once started);" in table
    for label, key in (("Safety coverage", "OUROBOROS_SAFETY_MODE"), ("Access level", "OUROBOROS_RUNTIME_MODE")):
        assert f"| {label} | `{defaults[key]}`" in table, label


def test_rating_publication_is_complete_and_source_bound():
    source = SITE / "public" / "llm-rating"
    generated = DOCS / "llm-rating"
    files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    assert files == {p.relative_to(generated) for p in generated.rglob("*") if p.is_file()}
    assert files
    for path in files:
        assert (source / path).read_bytes() == (generated / path).read_bytes(), f"rating mirrors differ: {path}"
    data = json.loads((source / "data/latest.json").read_text(encoding="utf-8"))
    assert data["status"] == "final"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["published"])
    archived = source / "data/editions" / data["edition"] / "edition.json"
    assert archived.read_bytes() == (source / "data/latest.json").read_bytes()
    if data["method_version"] >= 4:
        _assert_rating_column_reasons(data)
        _assert_rating_reviewers(data)
        # Keep the latest-payload safety check first, including when revision fields are missing.
        _assert_rating_public_text_has_no_amounts(data)
        _assert_rating_vote_source(data, source)
        _assert_rating_static_publication(source, data)
        if data["method_version"] >= 5:
            _assert_rating_views_and_questions(data)
    else:
        assert data["method_version"] == 3
        _assert_rating_per_kind_reasons(data)
    for path in (SITE / "index.html", DOCS / "index.html"):
        assert 'href="/llm-rating/"' in path.read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    if data["method_version"] >= 5:
        for mirror in (source, generated):
            _assert_rating_public_files_have_no_money(mirror, readme)
    _assert_rating_wrappers_and_asset_versions(source)
    if data["method_version"] >= 5:
        _assert_rating_published_counts(source, data, readme)
    for theme in ("light", "dark"):
        image = f"docs/llm-rating/card/latest-en-{theme}.png"
        assert image in readme and (REPO / image).is_file()
    if data["method_version"] >= 4:
        image = "docs/llm-rating/card/latest-en-light.png"
        alts = [tag.get("alt", "") for tag in _tags(readme, "img") if tag.get("src") == image]
        assert len(alts) == 1, "README must have one rating card img"
        assert re.search(r"\bcolumns?\b", alts[0], re.I) and not re.search(r"\bkinds?\s+of\s+work\b", alts[0], re.I), (
            "README rating card alt must describe columns, not kinds of work"
        )


def _assert_rating_assets_are_javascript(source, node):
    def check(label, *args, script=None):
        result = subprocess.run([node, "--check", *args], input=script, capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, f"Invalid published JavaScript: {label}\n{result.stderr}"

    for path in sorted((source / "assets").glob("*.js")):
        check(path.name, str(path))
    for path in sorted(source.rglob("*.html")):
        for i, script in enumerate(_rating_html(path.read_text(encoding="utf-8")).scripts, 1):
            check(f"{path.relative_to(source)} inline script {i}", "-", script=script)


def test_rating_published_javascript_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser execution is checked by the rating build")
    for mirror in (SITE / "public" / "llm-rating", DOCS / "llm-rating"):
        _assert_rating_assets_are_javascript(mirror, node)


def test_rating_javascript_check_detects_invalid_asset(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; cannot check the syntax control")
    assets = tmp_path / "assets"
    assets.mkdir()
    script = assets / "control.js"
    script.write_text("const ok = true;\n", encoding="utf-8")
    _assert_rating_assets_are_javascript(tmp_path, node)
    script.write_text("const broken = ;\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Invalid published JavaScript: control.js"):
        _assert_rating_assets_are_javascript(tmp_path, node)


def test_rating_november_archive_is_frozen_and_graded():
    archive = "data/editions/2026.11/edition.json"
    for mirror in (SITE / "public" / "llm-rating", DOCS / "llm-rating"):
        raw = (mirror / archive).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == NOVEMBER_SHA256, f"2026.11 archive SHA-256 changed: {mirror}"
        november = json.loads(raw)
        assert november["edition"] == "2026.11" and november["method_version"] == 3
        _assert_rating_per_kind_reasons(november)
        latest = json.loads((mirror / "data/latest.json").read_text(encoding="utf-8"))
        if latest["edition"] != "2026.11" and latest["method_version"] >= 4:
            _assert_rating_archive_share_head(mirror)


def test_rating_december_archive_is_frozen():
    for mirror in (SITE / "public" / "llm-rating", DOCS / "llm-rating"):
        _assert_rating_december_archive(mirror)


def _assert_rating_december_archive(mirror):
    """Edition 2026.12 is published: its data and its card stay byte for byte whichever edition is the latest, and once
    a later edition exists it has an archive page of its own."""
    raw = (mirror / "data/editions/2026.12/edition.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DECEMBER_SHA256, f"2026.12 archive SHA-256 changed: {mirror}"
    december = json.loads(raw)
    assert december["edition"] == "2026.12" and december["method_version"] == 4 and december["status"] == "final"
    _assert_rating_column_reasons(december)
    latest = json.loads((mirror / "data/latest.json").read_text(encoding="utf-8"))
    if latest["edition"] != "2026.12":
        _assert_rating_archive_share_head(mirror, "2026.12")


def _assert_rating_per_kind_reasons(data):
    assert any(cell["grade"] is not None for cell in data["cells"]), "a per-kind edition must carry graded cells"
    groups = {group["id"]: group for group in data["groups"]}
    works = {work["id"]: work for work in data["works"]}
    models = {model["id"]: model for model in data["models"]}
    main = {gid for gid, group in groups.items() if group["main"]}
    assert main
    rated = [m for m in models.values() if m.get("rank") is not None]
    assert rated
    for model in rated:
        for gid in main:
            assert model["groups"].get(gid, {}).get("grade") in RATING_GRADES, (model["id"], gid)
    for cell in data["cells"]:
        if cell["grade"] is None:
            continue
        reason = cell.get("note")
        if not reason:
            group = groups[works[cell["work"]]["group"]]
            assert group["main"], (cell["model"], cell["work"])
            axis = models[cell["model"]]["groups"][group["id"]]
            assert axis["grade"] == cell["grade"]
            reason = axis["reason"]
        assert all(reason.get(language, "").strip() for language in ("en", "ru"))


def _assert_rating_column_reasons(data):
    assert data["cells"] == [], "method >= 4 requires cells == []"
    main = {group["id"] for group in data["groups"] if group["main"]}
    assert len(main) == data["totals"]["columns"] > 0
    grounds = {"carried", "case", "cards", "role", "open", "coordinator"}
    # Method 5: a cell may carry a dash instead of a letter (no work of that kind with a known outcome). It has the
    # ground "nodata", only such a cell has it, and it still says in both languages what is missing.
    dash_allowed = data["method_version"] >= 5
    if dash_allowed:
        grounds = grounds | {"nodata"}
    rated = [model for model in data["models"] if model.get("rank") is not None]
    assert {model["id"] for model in rated} == {model["id"] for model in data["models"] if model["status"] == "rated"}
    assert len(rated) == data["totals"]["models"]["rated"] > 0
    cells = [model["groups"].get(gid, {}) for model in rated for gid in main]
    assert all("grade" in cell for cell in cells), "every main cell needs an explicit grade (null for a dash)"
    case_ids = {case["id"] for case in data["cases"]}
    letters = data["totals"]["letters"]
    assert letters["main"] == sum(1 for cell in cells if cell.get("grade") is not None) and letters.get("empty", 0) == sum(
        1 for cell in cells if cell.get("grade") is None
    ), "totals.letters disagrees with the letters and dashes of the table"
    for model in rated:
        assert main <= model["groups"].keys(), model["id"]
        if data["method_version"] >= 5:
            role = model.get("role")
            assert isinstance(role, dict) and all(isinstance(role.get(lang), str) and role[lang].strip() for lang in ("en", "ru")), (
                "method >= 5 requires a role tag", model["id"]
            )
        for gid in main:
            entry = model["groups"][gid]
            where = (model["id"], gid)
            assert entry.get("grade") in RATING_GRADES or (dash_allowed and entry.get("grade") is None), ("not a letter of the scale", *where)
            assert entry.get("ground") in grounds, ("unknown ground", *where)
            assert (entry.get("grade") is None) == (entry.get("ground") == "nodata"), ("a dash and the ground nodata go together", *where)
            if entry["ground"] == "case":
                links = entry.get("cases")
                assert isinstance(links, list) and links and all(cid in case_ids for cid in links), (
                    "ground=case requires existing case IDs", *where
                )
            for language in ("en", "ru"):
                reason = entry.get("reason", {}).get(language)
                assert isinstance(reason, str) and reason.strip(), (*where, language)
    if dash_allowed:
        _assert_rating_case_links(data, case_ids)
        compare = data.get("compare")
        if compare and compare.get("new_columns"):
            assert compare["comparable"] == compare["columns"] and not set(compare["comparable"]) & set(compare["new_columns"])


def _assert_rating_case_links(data, ids):
    def visit(node, path=()):
        if isinstance(node, dict):
            for key, val in node.items():
                where = (*path, key)
                if key == "cases" and where not in {("cases",), ("totals", "cases")}:
                    assert isinstance(val, list) and all(type(cid) is int and cid in ids for cid in val), (
                        "unknown public case reference", where, val
                    )
                else:
                    visit(val, where)
        elif isinstance(node, list):
            for i, val in enumerate(node):
                visit(val, (*path, i))

    visit(data)


def _assert_rating_reviewers(data):
    reviewers = data.get("reviewers")
    assert isinstance(reviewers, dict), "method >= 4 requires the reviewers object"
    rows = reviewers.get("rows")
    assert isinstance(rows, list) and rows, "reviewers.rows must be non-empty"
    families = {model["id"]: model["family"] for model in data["models"]}
    ids = families.keys()
    for row in rows:
        assert row["model"] in ids, ("reviewers.rows model", row["model"])
        _assert_rating_remark_counts(row, "checked")
    _assert_rating_remark_counts(reviewers["recheck"], "remarks")
    readers = reviewers["recheck"].get("readers")
    assert isinstance(readers, list) and len(readers) >= 2 and all(r in ids for r in readers), (
        "reviewer recheck requires at least two known readers", readers
    )
    original = {row["model"] for row in rows} | {reviewers.get("unbound", {}).get("requested")}
    reader_families = {families[r] for r in readers}
    assert len(set(readers)) == len(readers) and len(reader_families) >= 2, "reviewer recheck needs distinct readers from two families"
    assert reader_families.isdisjoint(families[r] for r in original & ids), "reviewer recheck readers cannot be from the original seats' families"
    for key, model_field, prose_fields in (
        ("advice", "who", ("text",)),
        ("examples", "models", ("title", "text")),
    ):
        entries = reviewers.get(key)
        assert isinstance(entries, list) and entries, f"reviewers.{key} must be non-empty"
        for entry in entries:
            models = entry.get(model_field)
            assert isinstance(models, list) and models and all(isinstance(mid, str) and mid in ids for mid in models), (key, models)
            for field in prose_fields:
                text = entry.get(field)
                assert isinstance(text, dict) and all(isinstance(text.get(lang), str) and text[lang].strip() for lang in ("en", "ru")), (key, field)


def _assert_rating_remark_counts(row, total):
    fields = ("confirmed", "open", "refuted", total)
    assert all(type(row.get(key)) is int and row[key] >= 0 for key in fields), ("reviewer counts", row)
    assert sum(row[key] for key in fields[:-1]) == row[total], ("inconsistent reviewer counts", row)


def _assert_rating_vote_verdict(item, models, columns, scale):
    counts = item["counts"]
    first, second = counts["first"], counts["second"]
    winner = None if first == second else int(second > first)
    actual = item.get("winner")
    assert "winner" in item and type(actual) is type(winner) and actual == winner, (
        "votes pair winner disagrees with counts", item["models"], actual, winner
    )
    column = item["column"]
    assert column is None or column in columns, ("votes pair must refer to a main column", column)
    letters = [m.get("groups", {}).get(column, {}).get("grade") for m in models]
    if None in letters:
        assert item.get("letters") is None, ("votes pair has letters without two ranked entries", item["models"], column)
        expected = "no_column"
    else:
        assert item.get("letters") == letters, ("votes pair letters disagree with ranking", item["models"], column)
        if letters[0] == letters[1]:
            expected = "same_letter"
        elif winner is None:
            expected = "no_majority"
        else:
            expected = "agree" if scale.index(letters[winner]) < scale.index(letters[1 - winner]) else "disagree"
    assert item.get("verdict") == expected, ("votes pair verdict disagrees with winner and ranking", item["models"], expected)


def _assert_rating_vote_source(data, source=None):
    revision = RATING_REVISION.fullmatch(data["edition"])
    if revision:
        assert data.get("revision_of") == revision[1], "a revision must have revision_of naming the edition it revises"
        _assert_rating_carried_blocks(data, source)
    else:
        _assert_rating_votes(data, source)


def _assert_rating_carried_blocks(data, source=None):
    """A revision casts no votes of its own: it carries the votes and the reviewers of the edition it revises,
    unchanged, and says above them which edition they were made for."""
    source = source or SITE / "public" / "llm-rating"
    revised = json.loads((source / "data/editions" / data["revision_of"] / "edition.json").read_text(encoding="utf-8"))
    assert revised["edition"] == data["revision_of"] == data["previous_edition"], "a revision revises the edition right before it"
    assert data["votes"] == revised["votes"], "a revision must carry the votes unchanged"
    assert data["reviewers"] == revised["reviewers"], "a revision must carry the reviewers unchanged"
    note = data.get("carried", {})
    assert note.get("from") == data["revision_of"], "the carried blocks must name the edition they were made for"
    _assert_rating_carried_fields(data, revised, note)
    if data.get("compare"):
        _assert_rating_comparable_columns(data, revised)
    text = note.get("note")
    assert isinstance(text, dict) and all(isinstance(text.get(lang), str) and text[lang].strip() for lang in ("en", "ru")), (
        "the carried blocks need their note in both languages"
    )
    _assert_carried_vote_letters(data, text)


def _assert_rating_carried_fields(data, revised, note):
    actual = note.get("what")
    assert isinstance(actual, list) and Counter(actual) == Counter(RATING_CARRIED_FIELDS), (
        "carried.what must list exactly the publication carry fields", actual, RATING_CARRIED_FIELDS
    )
    for field in RATING_CARRIED_FIELDS:
        fields = field.split(".")
        left, right = data, revised
        for key in fields:
            left, right = left[key], right[key]
        assert left == right, ("named carried field differs from the source edition", field)


def _assert_rating_comparable_columns(data, previous):
    current_cols = {g["id"] for g in data["groups"] if g["main"]}
    previous_cols = {g["id"] for g in previous["groups"] if g["main"]}
    common = current_cols & previous_cols
    unknown = set(data["compare"]["comparable"]) - common
    assert not unknown, ("compare.comparable names a column absent from one of its editions", sorted(unknown))


def _assert_carried_vote_letters(data, text):
    models = {model["id"]: model for model in data["models"]}
    groups = {group["id"]: group for group in data["groups"]}
    for pair in data["votes"]["pairs"]:
        column = pair["column"]
        if pair["letters"] is None:  # A carried no-column verdict has no letters to compare.
            continue
        for model_id, before in zip(pair["models"], pair["letters"], strict=True):
            model = models[model_id]
            after = model["groups"][column]["grade"]
            if before == after:
                continue
            # Accept only a disclosure of this particular model, column and transition in both languages.
            move = f"{before or '–'} → {after or '–'}"
            for lang in ("en", "ru"):
                fragments = re.split(r"[.!?;](?:\s+|$)", text[lang])
                assert any(all(part.casefold() in fragment.casefold() for part in (
                    model["short"], groups[column]["name"][lang], move,
                )) for fragment in fragments), ("carried vote letter differs from the current table", model_id, column, move, lang)


class _RatingHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.text = []
        self.metadata = []
        self.scripts = []
        self.inline = None
        self.title = None

    def handle_starttag(self, tag, attrs):
        if tag in {"head", "script", "style"}:
            self.hidden.append(tag)
        attrs = dict(attrs)
        if tag == "title" and "head" in self.hidden:
            self.title = []
        if tag == "meta" and "head" in self.hidden and attrs.get("content"):
            self.metadata.append(attrs["content"])
        if tag == "script" and "src" not in attrs:
            self.inline = []
        if not self.hidden and tag == "img" and "alt" in attrs:
            self.text.append(attrs["alt"])

    def handle_endtag(self, tag):
        if self.hidden and self.hidden[-1] == tag:
            self.hidden.pop()
        if tag == "title" and self.title is not None:
            self.metadata.append("".join(self.title))
            self.title = None
        if tag == "script" and self.inline is not None:
            self.scripts.append("".join(self.inline))
            self.inline = None

    def handle_data(self, data):
        if self.inline is not None:
            self.inline.append(data)
        if self.title is not None:
            self.title.append(data)
        if not self.hidden:
            self.text.append(data)


def _rating_html(html):
    parser = _RatingHTML()
    parser.feed(html)
    parser.close()
    return parser


def _assert_no_money(text, where):
    match = RATING_MONEY.search(unescape(text))
    assert not match, ("public rating text must name no sum of money or currency", where, match.group() if match else "")


def test_rating_head_keeps_only_title_and_meta_contents_for_money_scan():
    page = _rating_html('<head><title>25 GBP</title><meta name="description" content="15руб.">'
                        '<meta property="og:title" content="GBP 25"><meta name="twitter:title" content="25EUR">'
                        '<link href="x/25GBP"><script>"75 USD"</script></head><body>body</body>')
    assert page.text == ["body"]
    assert page.metadata == ["25 GBP", "15руб.", "GBP 25", "25EUR"]
    for text in page.metadata:
        with pytest.raises(AssertionError, match="public rating text must name no sum of money"):
            _assert_no_money(text, "head control")


def test_rating_no_column_vote_has_no_letters_to_compare():
    _assert_carried_vote_letters({"models": [], "groups": [], "votes": {"pairs": [
        {"column": None, "letters": None}, {"column": "review", "letters": None},
    ]}}, {})


def _assert_rating_public_text_has_no_amounts(data, where=()):
    if isinstance(data, dict):
        for key, value in data.items():
            _assert_rating_public_text_has_no_amounts(value, (*where, key))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            _assert_rating_public_text_has_no_amounts(value, (*where, i))
    elif isinstance(data, str):
        _assert_no_money(data, where)


def _assert_rating_public_files_have_no_money(source, readme):
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        label = str(path)
        if path.suffix == ".json":
            raw = path.read_bytes()
            if FROZEN_RATING_JSON.get(rel) == hashlib.sha256(raw).hexdigest():
                continue
            _assert_rating_public_text_has_no_amounts(json.loads(raw), (label,))
        elif path.name == "strings.js":
            prefix = "window.OUROBOROS_STRINGS="
            script = path.read_text(encoding="utf-8").strip()
            assert script.startswith(prefix) and script.endswith(";"), f"invalid rating strings wrapper: {label}"
            _assert_rating_public_text_has_no_amounts(json.loads(script[len(prefix):-1]), (label,))
        elif path.suffix == ".html":
            page = _rating_html(path.read_text(encoding="utf-8"))
            for text in (*page.text, *page.metadata):
                _assert_no_money(text, label)
        elif path.suffix == ".md":
            _assert_no_money(path.read_text(encoding="utf-8"), label)
    # Match the rating card by its exact target, not other pictures or unrelated prices in the root README.
    blocks = re.findall(r"<picture>.*?</picture>(?:\s*</a>\s*</p>\s*<p[^>]*>.*?</p>)?", readme, re.S)
    cards = [part for part in blocks if "docs/llm-rating/card/latest-en-light.png" in part]
    assert len(cards) == 1, "README must have one rating picture block"
    for text in _rating_html(cards[0]).text:
        _assert_no_money(text, "root README rating card")


def _rating_counts(data, columns):
    grades = [model["groups"][column]["grade"] for model in data["models"] if model.get("rank") is not None for column in columns]
    return sum(grade is not None for grade in grades), grades.count(None)


def _assert_rating_count_text(text, expected, where, *, required=True):
    matches = [("letters" if noun.startswith(("letter", "букв")) else "dashes", int(n))
               for n, noun in RATING_COUNT.findall(unescape(text).casefold())]
    assert not required or {kind for kind, _ in matches} == {"letters", "dashes"}, (
        "published rating counts missing", where, matches
    )
    wanted = dict(zip(("letters", "dashes"), expected))
    assert all(n == wanted[kind] for kind, n in matches), ("published rating counts disagree with table", where, matches, wanted)
    return bool(matches)


def _assert_rating_published_counts(source, data, readme):
    main = [g["id"] for g in data["groups"] if g["main"]]
    expected = _rating_counts(data, main)
    for filename in ("README.md", "METHOD.md", "CHANGELOG.md"):
        text = (source / filename).read_text(encoding="utf-8")
        if filename == "CHANGELOG.md":
            sections = re.split(r"(?m)^## (\S+)\s*$", text)
            text = dict(zip(sections[1::2], sections[2::2]))[data["edition"]]
        _assert_rating_count_text(text, expected, filename)
    for label, text in (("root README alt", re.findall(r'<img alt="([^"]*)" src="docs/llm-rating/card/latest-en-light\.png"', readme)),
                        ("root README caption", re.findall(r'(?m)^<p[^>]*><a [^>]*href="https://ouroboros-agent\.ai/llm-rating/"[^>]*><b>[^\n]*</p>$', readme))):
        assert len(text) == 1, ("published rating counts missing", label)
        _assert_rating_count_text(text[0], expected, label)
    archive_index = json.loads((source / "data/editions/index.json").read_text(encoding="utf-8"))
    latest_entry = next(e for e in archive_index["editions"] if e["edition"] == data["edition"])
    for lang in ("en", "ru"):
        _assert_rating_count_text(data["notes"]["changelog"][lang], expected, f"edition changelog {lang}")
        _assert_rating_count_text(latest_entry["changelog"][lang], expected, f"edition index {lang}")
    if new_columns := data.get("compare", {}).get("new_columns"):
        new_expected = _rating_counts(data, new_columns)
        previous = json.loads((source / "data/editions" / data["previous_edition"] / "edition.json").read_text(encoding="utf-8"))
        previous_expected = _rating_counts(previous, [g["id"] for g in previous["groups"] if g["main"]])
        for lang, previous_label in (("en", "earlier columns"), ("ru", "прежних колонок")):
            checked_new = False
            for i, item in enumerate(data["compare"]["bullets"]):
                text = item[lang]
                refers_to_previous = previous_label in text.casefold()
                matched = _assert_rating_count_text(text, previous_expected if refers_to_previous else new_expected,
                                                    f"compare bullet {i} {lang}", required=False)
                checked_new |= matched and not refers_to_previous
            assert checked_new, ("published rating counts missing", f"new columns {lang}")
    wrapped = (source / "assets/strings.js").read_text(encoding="utf-8").strip()
    counts = json.loads(wrapped.removeprefix("window.OUROBOROS_STRINGS=").removesuffix(";"))["card_table_count"]
    for lang, template in (("en", "{letters} letters · {dashes} dashes"), ("ru", "Буквы: {letters} · прочерки: {dashes}")):
        assert counts[lang] == template, ("published card count labels must read the letter and dash totals", lang, counts[lang])


def _assert_rating_views_and_questions(data):
    def bilingual(text):
        return isinstance(text, dict) and text.keys() == {"en", "ru"} and all(
            isinstance(text[lang], str) and text[lang].strip() for lang in ("en", "ru")
        )

    rated = {model["id"] for model in data["models"] if model["status"] == "rated"}
    main = [group["id"] for group in data["groups"] if group["main"]]
    views = data.get("views")
    assert isinstance(views, dict) and views.get("rows"), "method >= 5 requires the three views"
    view_ids = [column["id"] for column in views["columns"]]
    assert view_ids == ["quality", "now", "budget"], "the three views are quality, now and budget"
    for row in views["rows"]:
        assert row["model"] in rated and row.keys() == {"model", *view_ids}, ("a view row is one ranked model seen three ways", row.get("model"))
        assert all(bilingual(row[field]) for field in view_ids), "view rows are bilingual"
    assert [row["group"] for row in views["budget"]["rows"]] == main, "the budget block has one line per column of the main table"
    assert all(bilingual(row.get("text")) for row in views["budget"]["rows"]), "budget rows are bilingual"
    questions = data.get("faq")
    assert isinstance(questions, dict) and questions.get("items"), "method >= 5 requires readers' questions"
    assert all(item.keys() == {"q", "a"} for item in questions["items"]), "a reader's question is a question and an answer"
    assert all(bilingual(item[field]) for item in questions["items"] for field in ("q", "a")), "FAQ questions and answers are bilingual"


def _assert_rating_votes(data, source=None):
    votes = data.get("votes")
    assert isinstance(votes, dict), "method >= 4 requires the votes object"
    assert votes.get("source") == "final" and votes.get("complete") is True, "votes must be final and complete"
    models_by_id = {m["id"]: m for m in data["models"]}
    columns = {g["id"] for g in data["groups"] if g["main"]}
    scale = [tier["id"] for tier in data["scale"]]

    def bilingual(item, field):
        text = item.get(field)
        assert isinstance(text, dict) and all(
            isinstance(text.get(lang), str) and text[lang].strip() for lang in ("en", "ru")
        ), ("votes bilingual text", field, item)

    def count(item, field):
        value = item.get(field)
        assert type(value) is int and value >= 0, ("votes non-negative integer", field, item)
        return value

    item_totals = {}
    for key, labels in (("pairs", ("first", "second", "tie", "abstain")),
                        ("annotations", ("supported", "contradicted", "insufficient"))):
        items = votes.get(key)
        assert isinstance(items, list) and items, f"votes.{key} must be non-empty"
        assigned_total = counted_total = 0
        for item in items:
            bilingual(item, "description")
            if key == "pairs":
                models = item.get("models")
                assert isinstance(models, list) and len(models) == 2 and all(
                    isinstance(mid, str) and mid in models_by_id for mid in models
                ) and models[0] != models[1], ("votes pairs must name two known models", item)
            else:
                bilingual(item, "established")
            counts = item.get("counts")
            assert isinstance(counts, dict) and counts.keys() == set(labels), ("votes count labels", key, item)
            assigned, counted = count(item, "assigned"), sum(count(counts, label) for label in labels)
            assert counted <= assigned, ("votes item count exceeds assigned", key, item)
            if key == "pairs":
                _assert_rating_vote_verdict(item, [models_by_id[mid] for mid in models], columns, scale)
                if item["verdict"] == "disagree":
                    bilingual(item, "explain")
            assigned_total += assigned
            counted_total += counted
        item_totals[key] = (assigned_total, counted_total)

    voters = votes.get("voters")
    assert isinstance(voters, list) and voters, "votes.voters must be non-empty"
    counted_statuses = {"counted", "partially_counted"}
    returned_statuses = counted_statuses | {"excluded", "format_invalid", "evidence_unavailable", "model_mismatch", "duplicate_response"}
    statuses = returned_statuses | {"not_sent", "no_response", "access_unavailable", "model_error", "delivery_unknown"}
    status_counts, prior_table_counts = Counter(), Counter()
    seen_models = set()
    totals = {"with_pairs": 0, "with_annotations": 0}
    roster_tallies = Counter()
    for voter in voters:
        model, status, prior = voter.get("model"), voter.get("status"), voter.get("prior_table")
        assert model in models_by_id and model not in seen_models, ("votes voter must be a unique known model", voter)
        assert status in statuses and prior in {"seen", "not_detected", "unknown"}, ("votes voter status", voter)
        seen_models.add(model)
        status_counts[status] += 1
        prior_table_counts[prior] += 1
        counts = voter.get("counts")
        assert isinstance(counts, dict), ("votes voter counts", voter)
        for kind, suffix in (("pairwise", "pairs"), ("annotations", "annotations")):
            assigned, counted = (count(counts, kind + "_" + key) for key in ("assigned", "counted"))
            assert counted <= assigned, ("votes voter count exceeds assigned", voter)
            roster_tallies[(suffix, "assigned")] += assigned
            roster_tallies[(suffix, "counted")] += counted
            totals["with_" + suffix] += counted > 0
        assert (counts["pairwise_counted"] + counts["annotations_counted"] > 0) == (status in counted_statuses), (
            "votes voter counted status disagrees with counts", voter
        )

    rated = {m["id"] for m in data["models"] if m["status"] == "rated"}
    assert rated <= seen_models, ("votes roster must include every rated version", sorted(rated - seen_models))
    # Fable 5.1 is catalogued as historical context, but was not a voter in this edition.
    if data.get("revision_of", data["edition"]) == "2026.12":
        assert "claude-fable-5.1" not in seen_models, "Fable 5.1 is not a voter in 2026.12"
    if data.get("revision_of"):
        _assert_rating_carried_blocks(data, source)

    for key, pair in item_totals.items():
        assert pair == (roster_tallies[(key, "assigned")], roster_tallies[(key, "counted")]), (
            "votes tally disagrees with voter list", key, pair, roster_tallies
        )
    participation = votes.get("participation")
    assert isinstance(participation, dict), "votes.participation must be an object"
    totals.update(invited=len(voters), returned=sum(status_counts[s] for s in returned_statuses),
                  counted_ballots=sum(status_counts[s] for s in counted_statuses))
    for field, total in totals.items():
        assert count(participation, field) == total, ("votes participation disagrees with voter list", field, total)
    for field, expected in (("status_counts", status_counts), ("prior_table_counts", prior_table_counts)):
        value = participation.get(field)
        assert isinstance(value, dict), ("votes participation counts", field)
        assert {key: count(value, key) for key in value} == dict(expected), ("votes participation disagrees with voter list", field, expected)


def _assert_rating_archive_share_head(mirror, edition="2026.11"):
    month, card_sha256, words = RATING_ARCHIVES[edition]
    page = mirror / f"{edition}.html"
    assert page.is_file(), f"missing {month} archive share page"
    html = page.read_text(encoding="utf-8")
    head = html.split("</head>", 1)[0]
    tags = _tags(head)
    metas = _tags(head, "meta")
    url = f"{ORIGIN}/llm-rating/{edition}.html"
    card = f"card/editions/{edition}-en.png"
    digest = hashlib.sha256((mirror / card).read_bytes()).hexdigest()
    assert digest == card_sha256, f"{edition} archive PNG SHA-256 changed: {mirror}"
    digest = digest[:10]
    for file in ("index.html", "editions.html", f"{edition}.html"):
        bodies = _tags((mirror / file).read_text(encoding="utf-8"), "body")
        assert len(bodies) == 1
        cards = json.loads(bodies[0].get("data-archive-cards", "{}"))
        assert cards.get(edition) == digest, (f"{month} data-archive-cards must point to the frozen PNG", file)
    assert bodies[0].get("data-edition") == edition, f"{month} archive page must name its edition"
    assert [tag.get("href") for tag in tags if tag.get("rel") == "canonical"] == [url], f"{month} archive canonical URL"
    fields = [
        ("property", "og:url", url),
        ("property", "og:image", f"{ORIGIN}/llm-rating/{card}?v={digest}"),
        ("name", "twitter:image", f"{ORIGIN}/llm-rating/{card}?v={digest}"),
    ]
    for attr, key, expected in fields:
        assert [tag.get("content") for tag in metas if tag.get(attr) == key] == [expected], (f"{month} archive", key)
    for attr, key in (("name", "description"), ("property", "og:image:alt"), ("name", "twitter:image:alt")):
        values = [tag.get("content", "") for tag in metas if tag.get(attr) == key]
        assert len(values) == 1 and all(s in values[0] for s in words), (f"{month} archive", key)
    assert re.search(rf"<title>[^<]*Edition {re.escape(edition)}</title>", head), f"{month} archive title"
    fallback = re.findall(r"<noscript\b[^>]*>(.*?)</noscript>", html, flags=re.I | re.S)
    assert len(fallback) == 1, f"missing {month} no-JavaScript fallback"
    body = html.split("</head>", 1)[1]
    assert not re.search(r"data/latest\.(?:json|js)|card/(?:latest-|full-|og-)", body), (
        f"{month} no-JavaScript fallback must not link to latest files"
    )
    assert {tag.get("href") for tag in _tags(fallback[0], "a")} == {
        f"data/editions/{edition}/edition.json", card
    }, f"{month} no-JavaScript fallback must link frozen data and card"
    _assert_rating_relative_asset_versions(mirror, page.name, _tags(html))


def _assert_rating_static_publication(source, data):
    edition = data["edition"]
    book = json.loads((source / "data/editions/index.json").read_text(encoding="utf-8"))
    assert book["latest"] == edition, "edition index must name the final edition as latest"
    rows = {row["edition"]: row for row in book["editions"]}
    assert len(rows) == len(book["editions"]) and edition in rows, "edition index must list the final edition exactly once"
    on_disk = {p.parent.name for p in (source / "data/editions").glob("*/edition.json")}
    assert rows.keys() == on_disk, "edition index must cover all shipped editions"
    latest = rows[edition]
    for k in ("status", "method_version", "period", "published"):
        assert latest[k] == data[k], ("edition index metadata disagrees with final edition", k)
    totals = data["totals"]
    for k, expected in {"models_rated": totals["models"]["rated"], "work_types": totals["work_types"],
                        "columns": totals["columns"], "cases": totals["cases"],
                        "top": [m["short"] for m in data["models"] if m["status"] == "rated"][:3],
                        "changelog": data["notes"]["changelog"]}.items():
        assert latest[k] == expected, ("edition index metadata disagrees with final edition", k)
    assert latest["data"] == f"data/editions/{edition}/edition.json"
    assert latest["card"] == f"card/editions/{edition}-en.png" and (source / latest["card"]).is_file(), "edition index must link an existing final card"

    home = (source / "index.html").read_text(encoding="utf-8")
    archive = (source / "editions.html").read_text(encoding="utf-8")
    for html, tag, targets, message in (
        (home, "p", ["editions.html"], "static rating must link the editions list"),
        (archive, "ul", ["index.html" if eid == edition else f"{eid}.html" for eid in rows],
         "static editions list must link every edition page"),
    ):
        links = re.findall(rf'<{tag}\b[^>]*\bid="edition-links"[^>]*>(.*?)</{tag}>', html, re.S)
        assert len(links) == 1 and [a.get("href") for a in _tags(links[0], "a")] == targets, message
        assert all((source / target).is_file() for target in targets), message

    pairs = data["votes"]["pairs"]
    disagreements = [p for p in pairs if p["verdict"] == "disagree"]
    openings = re.findall(r'<p class="opening-vote">(.*?)</p>', home, re.S)
    assert len(openings) == 1, "opening vote summary must be present in static HTML"
    summary = openings[0]
    counts = re.search(r'<b>([^<]+)</b>', summary)
    assert counts and re.search(rf'\b{len(pairs):,} comparisons?, {len(disagreements):,} against my letters\b', unescape(counts[1])), (
        "static opening vote counts disagree with latest.json"
    )
    if disagreements:
        pair = disagreements[0]
        w, n = pair["winner"], 1 - pair["winner"]
        names = {m["id"]: m["short"] for m in data["models"]}
        column = next(g["name"]["en"] for g in data["groups"] if g["id"] == pair["column"])
        tally = (pair["counts"]["first"], pair["counts"]["second"])
        letters = pair["letters"]
        example = f"{column}: {names[pair['models'][w]]} — {tally[w]:,}:{tally[n]:,} against {names[pair['models'][n]]} (my letters: {letters[w]} and {letters[n]})."
        spans = re.findall(r'<span>([^<]+)</span>', summary)
        assert [unescape(text).strip() for text in spans] == [example], "static opening vote example disagrees with latest.json"
    assert [a.get("href") for a in _tags(summary, "a")] == ["#votes"], "static opening vote must link to #votes"
    assert any(tag.get("id") == "votes" for tag in _tags(home, "p")), "static opening vote needs a #votes destination"


def _assert_rating_wrappers_and_asset_versions(source):
    for wrapper, payload, variable in (
        ("data/latest.js", "data/latest.json", "OUROBOROS_RATING"),
        ("data/editions.js", "data/editions/index.json", "OUROBOROS_EDITIONS"),
    ):
        text = (source / wrapper).read_text(encoding="utf-8")
        prefix, suffix = f"window.{variable}=", ";\n"
        assert text.startswith(prefix) and text.endswith(suffix), wrapper
        assert json.loads(text[len(prefix):-len(suffix)]) == json.loads((source / payload).read_text(encoding="utf-8")), f"{wrapper} payload differs from {payload}"
    for page, data_js in (("index.html", "data/latest.js"), ("editions.html", "data/editions.js")):
        html = (source / page).read_text(encoding="utf-8")
        tags = _tags(html)
        preview = [tag for tag in tags if tag.get("property") == "og:image"]
        assert len(preview) == 1, page
        for language in ("en", "ru"):
            card = source / f"card/og-{language}.png"
            assert preview[0][f"data-v-{language}"] == hashlib.sha256(card.read_bytes()).hexdigest()[:10], (page, card)
        social = [tag["content"] for tag in tags if tag.get("property") == "og:image" or tag.get("name") == "twitter:image"]
        assert social == [f"{ORIGIN}/llm-rating/card/og-en.png?v={preview[0]['data-v-en']}"] * 2, page
        _assert_rating_relative_asset_versions(source, page, tags, data_js)


def _assert_rating_relative_asset_versions(source, page, tags, data_js=None):
    urls = [tag[attr] for tag in tags for attr in ("href", "src") if attr in tag]
    versioned = set()
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            continue
        params = parse_qs(parsed.query, keep_blank_values=True)
        if "v" in params:
            digest = hashlib.sha256((source / parsed.path.lstrip("/")).read_bytes()).hexdigest()[:10]
            assert params["v"] == [digest], f"{page}: {url} does not carry asset digest {digest}"
            versioned.add(parsed.path)
    required = {"assets/site.css", "assets/app.js", "assets/strings.js"}
    if data_js:
        required.add(data_js)
    assert required <= versioned, page


def _tags(page: str, name: str | None = None) -> list[dict[str, str]]:
    """The attributes of every ``<name>`` tag of a page, as an HTML parser reads them."""
    from html.parser import HTMLParser

    found: list[dict[str, str]] = []

    class Collector(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if name is None or tag == name:
                found.append({key: value or "" for key, value in attrs})

    Collector().feed(page)
    return found


def test_home_page_references_only_shipped_captures_and_a_hashed_social_card():
    """Every capture and font the home page or its stylesheet names ships byte-identically in site/public and in
    docs/, neither copy keeps an orphan, every capture on the source and on the served page declares its real pixel
    size, docs/ holds exactly the home bundles the served page loads, the served page is its source built, and every
    indexable page that names the social card, like its served copy, carries one og:image and one twitter:image whose
    cache-busting hash is the card's own content hash."""
    import hashlib

    from PIL import Image

    html = (SITE / "index.html").read_text(encoding="utf-8")
    served = (DOCS / "index.html").read_text(encoding="utf-8")
    css = (SITE / "src" / "home.css").read_text(encoding="utf-8")
    referenced = {image["src"][1:] for image in _tags(html, "img") if image.get("src", "").startswith("/ui/")}
    referenced |= {link["href"][1:] for link in _tags(html, "link") if link.get("href", "").startswith(("/fonts/", "/ui/"))}
    referenced |= set(re.findall(r"""url\(["']?/(fonts/[^)"']+)""", css))
    assert referenced
    for rel in sorted(referenced):
        assert (SITE / "public" / rel).is_file(), rel
        assert (SITE / "public" / rel).read_bytes() == (DOCS / rel).read_bytes(), rel
    for root in (SITE / "public", DOCS):
        captures = {p.relative_to(root).as_posix() for p in (root / "ui").glob("*.webp")}
        assert captures == {rel for rel in referenced if rel.startswith("ui/")}, root
        fonts = {p.relative_to(root).as_posix() for p in (root / "fonts").glob("*.woff2")}
        assert fonts == {rel for rel in referenced if rel.startswith("fonts/")}, root
        assert "SIL OPEN FONT LICENSE" in (root / "fonts" / "LICENSE-Inter.txt").read_text(encoding="utf-8")
    for page in (html, served):
        shown = [image for image in _tags(page, "img") if image.get("src", "").startswith("/ui/")]
        assert {image["src"][1:] for image in shown} == {rel for rel in referenced if rel.startswith("ui/")}
        for image in shown:
            with Image.open(SITE / "public" / image["src"][1:]) as capture:
                assert capture.size == (int(image.get("width") or 0), int(image.get("height") or 0)), image["src"]
    scripts = [tag.get("src", "") for tag in _tags(served, "script") if tag.get("type") == "module"]
    styles = [tag.get("href", "") for tag in _tags(served, "link") if tag.get("rel") == "stylesheet"]
    assert len(scripts) == 1 and re.fullmatch(r"/assets/home-[\w-]+\.js", scripts[0]), scripts
    assert len(styles) == 1 and re.fullmatch(r"/assets/home-[\w-]+\.css", styles[0]), styles
    assert {p.name for p in (DOCS / "assets").glob("home-*")} == {scripts[0].rsplit("/", 1)[1], styles[0].rsplit("/", 1)[1]}
    # the served page is this source built: apart from the two bundle lines the build rewrites, they read the same
    lines = [
        [line.strip() for line in page.splitlines() if line.strip() and "/src/home." not in line and "/assets/home-" not in line]
        for page in (html, served)
    ]
    assert lines[0] == lines[1]
    card = (REPO / "assets" / "og-preview.png").read_bytes()
    assert (DOCS / "assets" / "og-preview.png").read_bytes() == card
    expected = f"{ORIGIN}/assets/og-preview.png?v={hashlib.sha256(card).hexdigest()[:10]}"
    public = SITE / "public"
    pairs = [(source, DOCS / source.relative_to(public if public in source.parents else SITE)) for source in INDEXABLE.values()]
    naming = [pair for pair in pairs if any("og-preview.png" in page.read_text(encoding="utf-8") for page in pair)]
    assert (SITE / "index.html", DOCS / "index.html") in naming
    for pair in naming:
        for page in pair:
            metas = _tags(page.read_text(encoding="utf-8"), "meta")
            assert [meta.get("content") for meta in metas if meta.get("property") == "og:image"] == [expected], page
            assert [meta.get("content") for meta in metas if meta.get("name") == "twitter:image"] == [expected], page


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


# ---- Negative controls of the rating checks. Each new check must fail on the defect it guards, for its own reason:
# a check that passes whatever it is given proves nothing. The controls take the frozen files of the repository, break
# one thing in a private copy and require the named assertion; the unbroken copy must pass first.
def _rating_edition(edition, mirror=None):
    mirror = mirror or SITE / "public" / "llm-rating"
    return json.loads((mirror / "data/editions" / edition / "edition.json").read_text(encoding="utf-8"))


def _rating_cell(data, with_letter):
    """The first cell of the main table that has a letter, or the first that has a dash."""
    main = [group["id"] for group in data["groups"] if group["main"]]
    for model in data["models"]:
        for gid in main if model.get("rank") is not None else ():
            if (model["groups"][gid]["grade"] is not None) == with_letter:
                return model["groups"][gid]
    raise AssertionError("the edition has no such cell")


def _a_dash(data):
    _rating_cell(data, True)["grade"] = None
    data["totals"]["letters"] = {"main": data["totals"]["letters"]["main"] - 1, "empty": data["totals"]["letters"].get("empty", 0) + 1}


def _a_dash_dressed_as_nodata(data):
    _rating_cell(data, True)["ground"] = "nodata"
    _a_dash(data)


def _hide_fields_and_add_sum(data):
    for key in ("views", "faq", "carried"):
        del data[key]
    data["cases"][0]["text"]["ru"] = "Запуск обошёлся в 1 200,50 руб."


@pytest.mark.parametrize("amount", ["$15", "15$", "€ 20", "20 €", "14₽", "15 USD", "EUR 18.5", "5 руб.", "20 долл.", "15 долларов", "долларов 15", "40 евро", "15руб.", "GBP 25", "25 GBP", "GBP25", "коп. 30", "30 копеек"])
def test_rating_money_check_recognises_amounts_in_either_order(amount):
    _assert_rating_public_text_has_no_amounts({"en": "GPT-6.2, Qwen 3.8, 2026.12.1 and 2026-11-26", "ru": "396,9 млрд токенов"})
    with pytest.raises(AssertionError, match="name no sum of money"):
        _assert_rating_public_text_has_no_amounts({"nested": [{"text": amount}]})


RATING_DATA_CONTROLS = {
    # method 5: dashes, the count of letters, the role tag
    "dash-with-the-ground-of-a-letter": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, False).update(ground="role"),
                                          "a dash and the ground nodata go together"),
    "letter-with-the-ground-of-a-dash": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, True).update(ground="nodata"),
                                          "a dash and the ground nodata go together"),
    "dash-without-its-sentence": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, False)["reason"].update(ru=" "), "'ru'"),
    "letter-outside-the-scale": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, True).update(grade="D"), "not a letter of the scale"),
    "letters-miscounted": ("2026.12.1", _assert_rating_column_reasons, lambda d: d["totals"]["letters"].update(main=d["totals"]["letters"]["main"] + 1),
                           "totals.letters disagrees"),
    "dashes-uncounted": ("2026.12.1", _assert_rating_column_reasons, lambda d: d["totals"]["letters"].pop("empty"), "totals.letters disagrees"),
    "role-tag-missing": ("2026.12.1", _assert_rating_column_reasons, lambda d: next(m for m in d["models"] if m.get("rank") == 1).pop("role"),
                         "method >= 5 requires a role tag"),
    "role-tag-in-one-language": ("2026.12.1", _assert_rating_column_reasons, lambda d: next(m for m in d["models"] if m.get("rank") == 2)["role"].update(ru=""),
                                 "method >= 5 requires a role tag"),
    "dash-in-method-4": ("2026.12", _assert_rating_column_reasons, _a_dash, "not a letter of the scale"),
    "nodata-in-method-4": ("2026.12", _assert_rating_column_reasons, _a_dash_dressed_as_nodata, "not a letter of the scale"),
    # a revision carries the votes and the reviewers unchanged
    "carried-votes-edited": ("2026.12.1", _assert_rating_carried_blocks, lambda d: d["votes"]["counting_note"].update(en=d["votes"]["counting_note"]["en"] + " "),
                             "must carry the votes unchanged"),
    "carried-reviewers-edited": ("2026.12.1", _assert_rating_carried_blocks, lambda d: d["reviewers"]["rows"][0].update(open=d["reviewers"]["rows"][0]["open"] + 1),
                                 "must carry the reviewers unchanged"),
    "carried-from-another-edition": ("2026.12.1", _assert_rating_carried_blocks, lambda d: d["carried"].update({"from": "2026.11"}),
                                     "must name the edition they were made for"),
    "carried-note-in-one-language": ("2026.12.1", _assert_rating_carried_blocks, lambda d: d["carried"]["note"].update(ru=""), "need their note in both languages"),
    "revision-of-an-older-edition": ("2026.12.1", _assert_rating_carried_blocks, lambda d: d.update(revision_of="2026.11"), "revises the edition right before it"),
    # the three views, the budget block, readers' questions
    "views-removed": ("2026.12.1", _assert_rating_views_and_questions, lambda d: d.pop("views"), "requires the three views"),
    "view-of-an-unranked-model": ("2026.12.1", _assert_rating_views_and_questions, lambda d: d["views"]["rows"][0].update(model="gemini-4-hydrogen"),
                                  "one ranked model seen three ways"),
    "budget-line-missing": ("2026.12.1", _assert_rating_views_and_questions, lambda d: d["views"]["budget"]["rows"].pop(), "one line per column"),
    "answer-in-one-language": ("2026.12.1", _assert_rating_views_and_questions, lambda d: d["faq"]["items"][0]["a"].update(ru=""), "are bilingual"),
    "answer-deleted-in-one-language": ("2026.12.1", _assert_rating_views_and_questions, lambda d: d["faq"]["items"][0]["a"].pop("en"), "are bilingual"),
    "revision-of-missing": ("2026.12.1", _assert_rating_vote_source, lambda d: d.pop("revision_of"), "must have revision_of"),
    "explicit-dash-grade-missing": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, False).pop("grade"), "needs an explicit grade"),
    "case-id-unknown": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, True).update(ground="case", cases=[-1]), "requires existing case IDs"),
    "case-id-missing": ("2026.12.1", _assert_rating_column_reasons, lambda d: _rating_cell(d, True).update(ground="case", cases=[]), "requires existing case IDs"),
    "sum-in-case-outside-views": ("2026.12.1", _assert_rating_public_text_has_no_amounts, lambda d: d["cases"][0]["text"].update(en="About $15 a turn."), "name no sum of money"),
    "sum-in-words": ("2026.12.1", _assert_rating_public_text_has_no_amounts, lambda d: d["faq"]["items"][1]["a"].update(ru="Около 15 долларов."), "name no sum of money"),
    "sum-with-three-fields-deleted": ("2026.12.1", _assert_rating_public_text_has_no_amounts, _hide_fields_and_add_sum, "name no sum of money"),
}


@pytest.mark.parametrize("control", sorted(RATING_DATA_CONTROLS))
def test_rating_data_checks_fail_on_the_defect_they_guard(control):
    edition, check, break_it, message = RATING_DATA_CONTROLS[control]
    check(_rating_edition(edition))            # the published edition itself passes
    broken = _rating_edition(edition)
    break_it(broken)
    with pytest.raises(AssertionError, match=re.escape(message)):
        check(broken)


def _replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path.name, old, text.count(old))
    path.write_text(text.replace(old, new), encoding="utf-8")


RATING_ARCHIVE_CONTROLS = {
    "data-edited": (lambda m: (m / "data/editions/2026.12/edition.json").write_bytes((m / "data/editions/2026.12/edition.json").read_bytes() + b" "),
                    "2026.12 archive SHA-256 changed"),
    "card-swapped": (lambda m: shutil.copyfile(m / "card/editions/2026.11-en.png", m / "card/editions/2026.12-en.png"), "2026.12 archive PNG SHA-256 changed"),
    "page-removed": (lambda m: (m / "2026.12.html").unlink(), "missing December archive share page"),
    "page-of-another-edition": (lambda m: _replace_once(m / "2026.12.html", 'data-edition="2026.12"', 'data-edition="2026.12.1"'),
                                "December archive page must name its edition"),
    "card-digest-on-the-rating-page": (lambda m: _replace_once(m / "index.html", f"&quot;2026.12&quot;:&quot;{DECEMBER_CARD_SHA256[:10]}", "&quot;2026.12&quot;:&quot;0000000000"),
                                       "December data-archive-cards must point to the frozen PNG"),
    "title-of-the-latest-edition": (lambda m: _replace_once(m / "2026.12.html", "Edition 2026.12</title>", "Edition 2026.12.1</title>"), "December archive title"),
    "canonical-of-the-latest-page": (lambda m: _replace_once(m / "2026.12.html", f'<link rel="canonical" href="{ORIGIN}/llm-rating/2026.12.html">',
                                                           f'<link rel="canonical" href="{ORIGIN}/llm-rating/">'), "December archive canonical URL"),
    "fallback-links-the-latest-data": (lambda m: (m / "2026.12.html").write_text(
        (m / "2026.12.html").read_text(encoding="utf-8").replace("data/editions/2026.12/edition.json", "data/latest.json"), encoding="utf-8"),
        "December no-JavaScript fallback must not link to latest files"),
}


@pytest.mark.parametrize("control", sorted(RATING_ARCHIVE_CONTROLS))
def test_rating_december_archive_check_fails_on_the_defect_it_guards(control, tmp_path):
    break_it, message = RATING_ARCHIVE_CONTROLS[control]
    mirror = tmp_path / "llm-rating"
    shutil.copytree(SITE / "public" / "llm-rating", mirror)
    _assert_rating_december_archive(mirror)    # the published copy itself passes
    break_it(mirror)
    with pytest.raises(AssertionError, match=re.escape(message)):
        _assert_rating_december_archive(mirror)
