from pathlib import Path

from ouroboros.tools.scope_review_contract import COUPLING_QUESTION_IDS


ROOT = Path(__file__).resolve().parents[1]


def test_pull_request_template_has_one_universal_agent_review_block():
    template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
        encoding="utf-8"
    )

    assert "The PR base branch is `ouroboros`" in template
    assert "I did **not** bump `VERSION`" in template
    assert template.count("## Review evidence") == 1
    assert "Authoring agent/context" in template
    assert "Separate review agent/context" in template
    assert "Reviewer model and effort (when exposed)" in template
    assert "Reviewed base SHA" in template
    assert "Reviewed head SHA" in template
    assert "Findings and disposition" in template
    assert "coverage limitations" in template
    assert "PASS`, `NEEDS_CHANGES`, `INCOMPLETE`, or `NOT_RUN" in template
    assert "If not run, reason" in template
    # 21=A: the checklist coverage travels as a markdown table plus the raw
    # reviewer JSON, both inside the single review-evidence block — every row,
    # not just the endpoints.
    for item in COUPLING_QUESTION_IDS:
        assert template.count(f"| {item} |") == 1
    assert "Reviewer checklist JSON" in template
    assert "scripts/validate_scope_receipt.py" in template
    assert "self-review in the\nauthoring conversation does not" in template
    assert "Agent assistance (optional)" not in template
    assert "Human verification" not in template
    assert "Triad verdict" not in template
    assert "Scope verdict" not in template
    assert "OpenRouter" not in template


def test_missing_session_model_is_labelled_absent_in_evidence():
    from scripts.contributor_review_evidence import _session_evidence

    receipt = {"model_verification": "not_requested"}
    _session_evidence(
        surface="triad", slot_id="slot-1",
        route={"target_id": "codex=gpt-5.6-sol"}, status="responded",
        observed_model="", usage={"delegated_route": "codex"}, transcript="",
        deltas=[], receipt=receipt, mismatches=[],
    )

    assert receipt["model_verification"] == "absent"


def test_manual_ci_keeps_readonly_test_jobs_and_explicit_release_permissions():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    quick_job = workflow.partition("\n  quick-test:\n")[2].partition(
        "\n  # ──────────────────────────────────────────────────────────────────"
    )[0]

    assert "\n  pull_request:" not in workflow
    assert "\n  workflow_dispatch:" in workflow
    assert "\n  pull_request_target:" not in workflow
    # Dormant job predicates retain their secret boundaries even with PR/schedule
    # triggers disabled; a ref alone must never enable the paid provider job.
    assert "github.event_name != 'schedule'" in workflow.partition(
        "\n  integration-test:\n")[2].partition("\n    uses:")[0]
    assert "permissions:\n  contents: read" in workflow
    # No job keeps a predicate for the removed pull-request trigger.
    assert "github.base_ref == 'ouroboros'" not in workflow
    assert "secrets." not in quick_job
    assert "release:\n" in workflow
    assert "      contents: write" in workflow


def test_trusted_provider_ci_wires_full_secret_policy_and_release_dependency():
    workflows = ROOT / ".github" / "workflows"
    workflow = (workflows / "ci.yml").read_text(encoding="utf-8")
    # The canary steps live in one reusable file; ci.yml calls it for manual
    # runs and tags, the standalone wrapper only on explicit dispatch.
    shared = (workflows / "provider-canary.yml").read_text(encoding="utf-8")
    push = (workflows / "provider-canary-push.yml").read_text(encoding="utf-8")
    integration_job = workflow.partition("\n  integration-test:\n")[2].partition(
        "\n  # ──────────────────────────────────────────────────────────────────"
    )[0]
    release = workflow.partition("\n  release:\n")[2]

    assert integration_job
    assert "uses: ./.github/workflows/provider-canary.yml" in integration_job
    assert "github.event_name == 'pull_request'" not in integration_job
    assert "github.event_name == 'workflow_dispatch'" in integration_job
    assert "refs/tags/v" in integration_job
    assert "refs/heads/" not in integration_job
    assert "\n  push:" not in push
    assert "\n  workflow_dispatch:" in push
    assert "uses: ./.github/workflows/provider-canary.yml" in push
    assert "pull_request" not in push

    for secret in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MINIMAX_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
    ):
        for text in (integration_job, shared, push):
            assert f"{secret}: ${{{{ secrets.{secret} }}}}" in text

    assert " -rs " in shared
    # The release job itself requires the canaries (the preflight checks only the tag).
    assert "needs.integration-test.result == 'success'" in release


def test_repository_has_explicit_mit_license_holder():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Anton Razzhigaev" in license_text
    assert "Andrew Kaznacheev" not in license_text


def test_scope_receipt_validator_reuses_the_runtime_contract(tmp_path, capsys):
    """The contributor-facing validator is a thin CLI over
    normalize_scope_items — a receipt it accepts matches what the project's
    own scope reviewers must produce (10=A)."""
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from scripts.validate_scope_receipt import main

    good = [
        {"item": item, "verdict": "PASS", "severity": "advisory",
         "reason": f"Checked {item} against the concrete diff artifacts involved."}
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ]
    good_path = tmp_path / "good.json"
    good_path.write_text(__import__("json").dumps(good), encoding="utf-8")
    assert main(["validate", str(good_path)]) == 0
    assert "valid:" in capsys.readouterr().out

    missing = good[:-1]
    bad_path = tmp_path / "missing.json"
    bad_path.write_text(__import__("json").dumps(missing), encoding="utf-8")
    assert main(["validate", str(bad_path)]) == 1
    assert "missing required items" in capsys.readouterr().err

    dup = good + [dict(good[0])]
    dup_path = tmp_path / "dup.json"
    dup_path.write_text(__import__("json").dumps(dup), encoding="utf-8")
    assert main(["validate", str(dup_path)]) == 1
    assert "duplicate PASS" in capsys.readouterr().err

    not_json = tmp_path / "broken.json"
    not_json.write_text("{nope", encoding="utf-8")
    assert main(["validate", str(not_json)]) == 1
    assert "no JSON array" in capsys.readouterr().err


def test_scope_receipt_validator_accepts_fenced_and_embedded_arrays(tmp_path, capsys):
    """CONTRIBUTING's compact instruction asks for the array plus surrounding
    narrative — the validator must read what the runtime scope parser would
    read (extract_json_array), not only a bare file."""
    import json

    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from scripts.validate_scope_receipt import main

    rows = json.dumps([
        {"item": item, "verdict": "PASS", "severity": "advisory",
         "reason": f"Checked {item} against the concrete diff artifacts involved."}
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ])
    fenced = tmp_path / "fenced.md"
    fenced.write_text(
        "Reviewer notes before the receipt.\n\n```json\n" + rows + "\n```\n\nVerdict: PASS\n",
        encoding="utf-8",
    )
    assert main(["validate", str(fenced)]) == 0
    assert "valid:" in capsys.readouterr().out

    prose_only = tmp_path / "prose.md"
    prose_only.write_text("No receipt here at all.", encoding="utf-8")
    assert main(["validate", str(prose_only)]) == 1
    assert "no JSON array" in capsys.readouterr().err


def test_scope_receipt_validator_cli_edges(tmp_path, capsys):
    from scripts.validate_scope_receipt import main

    assert main(["validate", "--help"]) == 0
    assert "Usage" in capsys.readouterr().out

    assert main(["validate", str(tmp_path / "absent.json")]) == 1
    assert "cannot read receipt" in capsys.readouterr().err


def test_scope_receipt_survives_trailing_markdown_checkboxes(tmp_path, capsys):
    import json
    from scripts.validate_scope_receipt import main
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from ouroboros.triad_review import extract_json_array

    rows = [{"item": item, "verdict": "FAIL" if i == 0 else "PASS", "severity": "advisory",
             "reason": f"Checked the concrete {item} source and its consumers."}
            for i, item in enumerate(sorted(SCOPE_REQUIRED_ITEMS))]
    path = tmp_path / "receipt.md"
    for prefix, suffix in [("", "\n- [ ] Follow up"), ("- [ ] Before\n", "\n[]\n- [ ] After"), ("```json\n", "\n```\n- [ ] Remaining")]:
        path.write_text(prefix + json.dumps(rows) + suffix)
        assert main(["validate", str(path)]) == 0
        assert "1 FAIL row(s)" in capsys.readouterr().out
    # Whole JSON objects are not repaired by selecting a nested valid array.
    path.write_text(json.dumps({"nested": rows}))
    assert main(["validate", str(path)]) == 1
    path.write_text("- [ ] Only a checkbox")
    assert main(["validate", str(path)]) == 1
    assert extract_json_array("[]") == []
