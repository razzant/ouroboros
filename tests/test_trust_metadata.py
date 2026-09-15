from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from tests._governance_docs_shared import architecture_text

REPO = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "6.92.1"
RELEASE_TAG = f"v{RELEASE_VERSION}"
RELEASE_COMMIT = "d8087f42e4d369db393cbfb8318b8a82f9bb73b2"


def test_code_of_conduct_is_customized_contributor_covenant_3():
    text = (REPO / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")

    assert text.startswith("# Contributor Covenant 3.0 Code of Conduct\n")
    assert "https://t.me/abstractDL" in text
    assert "[NOTE:" not in text
    assert "Contributor Covenant, version 3.0" in text
    assert "CC BY-SA 4.0" in text


def test_citation_metadata_has_the_approved_author_and_acknowledgment():
    citation = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "Ouroboros"
    assert citation["type"] == "software"
    assert citation["authors"] == [{"family-names": "Razzhigaev", "given-names": "Anton"}]
    assert citation["repository-code"] == "https://github.com/razzant/ouroboros"
    assert citation["url"] == "https://ouroboros-agent.ai/"
    assert citation["license"] == "MIT"
    assert "The Ouroboros agent contributed" in citation["abstract"]
    forbidden = {"orcid", "affiliation", "version", "date-released", "doi"}
    assert forbidden.isdisjoint(citation)
    assert forbidden.isdisjoint(citation["authors"][0])
    preferred = citation["preferred-citation"]
    assert preferred == {
        "type": "article",
        "title": "Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution",
        "authors": [
            {"family-names": "Razzhigaev", "given-names": "Anton"},
            {"family-names": "Gritsaev", "given-names": "Andrei"},
            {"family-names": "Kaznacheev", "given-names": "Andrei"},
            {"family-names": "Dragunov", "given-names": "Nikita"},
            {"family-names": "Yampolskiy", "given-names": "Roman"},
            {"family-names": "Kuznetsov", "given-names": "Andrei"},
        ],
        "journal": "arXiv",
        "year": 2026,
        "doi": "10.48550/arXiv.2608.08311",
        "url": "https://arxiv.org/abs/2608.08311",
    }


def test_scorecard_workflow_is_fully_pinned_and_publishes_sarif():
    path = REPO / ".github" / "workflows" / "scorecard.yml"
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["on"] == {
        "push": {"branches": ["main"]},
        "schedule": [{"cron": "30 1 * * 6"}],
    }
    assert workflow["permissions"] == "read-all"
    job = workflow["jobs"]["analysis"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {
        "security-events": "write",
        "id-token": "write",
    }

    ordered_steps = job["steps"]
    assert [(step["name"], step["uses"]) for step in ordered_steps] == [
        (
            "Checkout code",
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        ),
        (
            "Run analysis",
            "ossf/scorecard-action@2d1146689b8cda280b9bc96326124645441f03bc",
        ),
        (
            "Upload artifact",
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        ),
        (
            "Upload to code-scanning",
            "github/codeql-action/upload-sarif@f205ea1c3313d32999d8d6a48b4f6530d4437b38",
        ),
    ]
    steps = {step["name"]: step for step in ordered_steps}

    assert steps["Checkout code"]["with"] == {"persist-credentials": "false"}
    scorecard = steps["Run analysis"]["with"]
    assert scorecard == {
        "results_file": "results.sarif",
        "results_format": "sarif",
        "publish_results": "true",
    }
    assert steps["Upload artifact"]["with"] == {
        "name": "SARIF file",
        "path": "results.sarif",
        "retention-days": "5",
    }
    assert steps["Upload to code-scanning"]["with"] == {"sarif_file": "results.sarif"}


def test_benchmark_evidence_is_release_bound_and_excludes_unready_evidence():
    path = REPO / "docs" / "benchmarks" / "evidence.json"
    raw = path.read_text(encoding="utf-8")
    evidence = json.loads(raw)

    assert "gaia" not in raw.lower()
    assert evidence["schema_version"] == 1
    assert evidence["snapshot"] == {
        "repository": "https://github.com/razzant/ouroboros",
        "version": RELEASE_VERSION,
        "tag": RELEASE_TAG,
        "commit": RELEASE_COMMIT,
        "claims_source": (f"https://github.com/razzant/ouroboros/blob/{RELEASE_COMMIT}/README.md#benchmarks"),
    }
    rows = evidence["benchmarks"]
    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["reporting"]["self_reported"] is True for row in rows)
    assert all(row["reporting"]["evidence_status"] for row in rows)


def test_benchmark_evidence_keeps_exact_scores_and_hugging_face_revisions():
    evidence = json.loads((REPO / "docs" / "benchmarks" / "evidence.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in evidence["benchmarks"]}

    expected_scores = {
        "terminal-bench-2.1-claude-opus-5-high": 86.74,
        "terminal-bench-2.1-claude-opus-4.8-high": 80.22,
        "terminal-bench-2.1-gpt-5.5": 84.3,
        "terminal-bench-2.1-grok-4.5": 84.94,
        "osworld-verified-claude-opus-5": 90.69,
        "osworld-verified-claude-sonnet-4.6": 83.27,
        "cl-bench-claude-sonnet-4.6": 0.2301,
        "swe-bench-pro-gpt-5.6-luna": 58.2,
    }
    assert {key: row["result"]["value"] for key, row in rows.items()} == expected_scores
    assert rows["terminal-bench-2.1-claude-opus-5-high"]["result"]["raw_value"] == 86.97
    assert rows["terminal-bench-2.1-gpt-5.5"]["result"]["calculation"] == {
        "self_reported_aggregation": "mean_over_tasks_of_per_task_pass_rate",
        "self_reported_unrounded_value": 84.26966292134831,
        "tasks": 89,
        "scheduled_trials": 445,
        "graded_trials": 444,
        "passed_trials": 374,
        "ungraded_trials_excluded": 1,
        "ungraded_trial": {
            "task": "mcmc-sampling-stan",
            "graded_task_passes": 4,
            "graded_task_trials": 4,
        },
        "leaderboard_compatible": {
            "aggregation": "passed_trials_over_scheduled_trials_with_errors_counted_as_zero",
            "unrounded_value": 84.04494382022472,
            "rounded_value": 84.04,
            "display_value": 84.0,
        },
    }
    assert rows["terminal-bench-2.1-gpt-5.5"]["reporting"] == {
        "self_reported": True,
        "submission_status": "open",
        "submission_validation": "failed_source_filter",
        "evidence_status": "public_run",
    }
    assert rows["cl-bench-claude-sonnet-4.6"]["result"]["rank"] == 1
    assert rows["swe-bench-pro-gpt-5.6-luna"]["analysis"] == {
        "test": "McNemar",
        "p_value": 0.4,
        "conclusion": "no_significant_difference",
    }

    expected_revisions = {
        "razzant/ouroboros-osworld-verified-opus5": ("f52ebf2248ce0ce0c496db18f5e6edce631304fa"),
        "razzant/ouroboros-osworld-verified-sonnet46": ("0e8ad516a4eeaa586607ead400429885814e7633"),
        "razzant/ouroboros-clbench-traces": ("85958a21989ee7a52efaaf6d26d498f831835c70"),
        "razzant/swepro-luna-matched-pair": ("62280378fd82ab9df0b7216745cd42e559ab3435"),
    }
    found_revisions = {}
    for row in rows.values():
        for item in row["evidence"]:
            assert item["url"].startswith("https://")
            if item["kind"] == "hugging_face_dataset":
                assert re.fullmatch(r"[0-9a-f]{40}", item["revision"])
                assert item["url"].endswith(f"/tree/{item['revision']}")
                found_revisions[item["repository"]] = item["revision"]
    assert found_revisions == expected_revisions


def test_architecture_registers_each_new_public_metadata_surface():
    architecture = architecture_text(REPO)

    for path in (
        ".github/workflows/scorecard.yml",
        "CODE_OF_CONDUCT.md",
        "CITATION.cff",
        "docs/benchmarks/evidence.json",
        "site/paper/index.html",
    ):
        assert path in architecture
    assert "README remains the claim SSOT" in architecture
