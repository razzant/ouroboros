from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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
NOVEMBER_SHA256 = "9976660610a53f0352a8df529943653de20d17f605741fe47a8ffc3fa8345709"
NOVEMBER_CARD = "card/editions/2026.11-en.png"
NOVEMBER_CARD_SHA256 = "15d6fbada68b212adcd9e483b99a29a1bf0811e104617a5f898030f21b2db779"
RATING_GRADES = {"S", "A", "B", "C"}
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
        _assert_rating_votes(data)
        _assert_rating_static_publication(source, data)
    else:
        assert data["method_version"] == 3
        _assert_rating_per_kind_reasons(data)
    _assert_rating_wrappers_and_asset_versions(source)
    for path in (SITE / "index.html", DOCS / "index.html"):
        assert 'href="/llm-rating/"' in path.read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
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
    rated = [model for model in data["models"] if model.get("rank") is not None]
    assert {model["id"] for model in rated} == {model["id"] for model in data["models"] if model["status"] == "rated"}
    assert len(rated) == data["totals"]["models"]["rated"] > 0
    for model in rated:
        assert main <= model["groups"].keys(), model["id"]
        for gid in main:
            entry = model["groups"][gid]
            where = (model["id"], gid)
            assert entry.get("grade") in RATING_GRADES, where
            assert entry.get("ground") in grounds, where
            for language in ("en", "ru"):
                reason = entry.get("reason", {}).get(language)
                assert isinstance(reason, str) and reason.strip(), (*where, language)


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


def _assert_rating_votes(data):
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
    if data["edition"] == "2026.12":
        assert "claude-fable-5.1" not in seen_models, "Fable 5.1 is not a voter in 2026.12"

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


def _assert_rating_archive_share_head(mirror):
    page = mirror / "2026.11.html"
    assert page.is_file(), "missing November archive share page"
    html = page.read_text(encoding="utf-8")
    head = html.split("</head>", 1)[0]
    tags = _tags(head)
    metas = _tags(head, "meta")
    url = f"{ORIGIN}/llm-rating/2026.11.html"
    card = NOVEMBER_CARD
    digest = hashlib.sha256((mirror / card).read_bytes()).hexdigest()
    assert digest == NOVEMBER_CARD_SHA256, f"2026.11 archive PNG SHA-256 changed: {mirror}"
    digest = digest[:10]
    for file in ("index.html", "editions.html", "2026.11.html"):
        bodies = _tags((mirror / file).read_text(encoding="utf-8"), "body")
        assert len(bodies) == 1
        cards = json.loads(bodies[0].get("data-archive-cards", "{}"))
        assert cards.get("2026.11") == digest, ("November data-archive-cards must point to the frozen PNG", file)
    assert [tag.get("href") for tag in tags if tag.get("rel") == "canonical"] == [url], "November archive canonical URL"
    fields = [
        ("property", "og:url", url),
        ("property", "og:image", f"{ORIGIN}/llm-rating/{card}?v={digest}"),
        ("name", "twitter:image", f"{ORIGIN}/llm-rating/{card}?v={digest}"),
    ]
    for attr, key, expected in fields:
        assert [tag.get("content") for tag in metas if tag.get(attr) == key] == [expected], ("November archive", key)
    for attr, key in (("name", "description"), ("property", "og:image:alt"), ("name", "twitter:image:alt")):
        values = [tag.get("content", "") for tag in metas if tag.get(attr) == key]
        assert len(values) == 1 and all(s in values[0] for s in ("Edition 2026.11", "11 models", "32 kinds of work")), ("November archive", key)
    assert re.search(r"<title>[^<]*Edition 2026\.11</title>", head), "November archive title"
    fallback = re.findall(r"<noscript\b[^>]*>(.*?)</noscript>", html, flags=re.I | re.S)
    assert len(fallback) == 1, "missing November no-JavaScript fallback"
    body = html.split("</head>", 1)[1]
    assert not re.search(r"data/latest\.(?:json|js)|card/(?:latest-|full-|og-)", body), (
        "November no-JavaScript fallback must not link to latest files"
    )
    assert {tag.get("href") for tag in _tags(fallback[0], "a")} == {
        "data/editions/2026.11/edition.json", card
    }, "November no-JavaScript fallback must link frozen data and card"
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
