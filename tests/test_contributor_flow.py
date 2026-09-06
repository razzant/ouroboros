from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_contributor_flow_is_agent_first_and_route_neutral():
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "against lowercase `ouroboros`, not `main` or `ouroboros-stable`" in guide
    assert "Coding agents and people must read [CONTRIBUTING.md]" in readme
    for path in (
        "BIBLE.md", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md",
        "docs/CHECKLISTS.md",
    ):
        assert path in guide
    # 11=A: only the checklist SSOT is read cover-to-cover; the other project
    # documents are mapped by headings with relevant sections read in full.
    assert "[`docs/CHECKLISTS.md`](docs/CHECKLISTS.md)" in guide
    assert "**in full**" in guide
    assert "navigation map" in guide
    assert "read every section relevant" in guide
    # Negative pins: the retired blanket contract must not resurface anywhere
    # in the two entry documents (README routes contributors here).
    assert "read these files **in full**" not in guide
    assert "The required project documents were read in full" not in guide
    assert "in full before editing" not in readme.replace(
        "reading every section relevant to your change in full before editing", "")
    # 10=A: the agentic checklist review is the main path and must cover all
    # eight Intent/Scope items with the runtime output contract.
    assert "agentic checklist review" in guide
    for item in (
        "intent_alignment", "forgotten_touchpoints", "cross_surface_consistency",
        "regression_surface", "prompt_doc_sync", "architecture_fit",
        "cross_module_bugs", "implicit_contracts",
    ):
        assert item in guide
    assert "scripts/validate_scope_receipt.py" in guide
    assert "separate agent context" in guide
    assert "Reviewing in the authoring conversation does not count" in guide
    assert "Mark the review `NOT_RUN`" in guide
    # The script lane is honest about its budget shape instead of failing
    # contributors by surprise (#395).
    assert "maintainer / large-window tooling" in guide
    assert "SCOPE_REVIEW_BLOCKED" in guide
    # The honest budget-shape paragraph names the full required pack and the
    # session route's own window requirement (sol round-1).
    assert "prompts, contracts, canonical docs" in guide
    assert "confirmed 200K+ window" in guide
    assert "SHAPE, not truth" in guide
    assert "--contributor" in guide
    assert "--base-ref upstream/ouroboros" in guide
    assert "--head-ref HEAD" in guide
    assert "review-packet.zip" in guide
    assert "evidence, not a promise to merge" in guide
    assert "OpenRouter" not in guide


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
    # reviewer JSON, both inside the single review-evidence block — all eight
    # rows, not just the endpoints.
    for item in (
        "intent_alignment", "forgotten_touchpoints", "cross_surface_consistency",
        "regression_surface", "prompt_doc_sync", "architecture_fit",
        "cross_module_bugs", "implicit_contracts",
    ):
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


def test_pull_request_ci_is_fork_safe_and_does_not_enable_provider_jobs():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    quick_job = workflow.partition("\n  quick-test:\n")[2].partition(
        "\n  # ──────────────────────────────────────────────────────────────────"
    )[0]

    assert "pull_request:\n    branches: [ouroboros]" in workflow
    assert "\n  pull_request_target:" not in workflow
    # A schedule IS admitted now (owner 9A: the keyless system-e2e-mock lane),
    # so what this used to say by banning the trigger outright it now says
    # directly — the unattended run must not reach the paid provider job. Its
    # branch conditions match the default branch ref a cron run carries, hence
    # the explicit event guard, which has to come FIRST to gate the whole `||`.
    assert "github.event_name != 'schedule'" in workflow.partition(
        "\n  integration-test:\n")[2].partition("\n    runs-on:")[0]
    assert "permissions:\n  contents: read" in workflow
    assert "github.event_name == 'pull_request' && github.base_ref == 'ouroboros'" in workflow
    assert "secrets." not in quick_job
    assert "release:\n" in workflow
    assert "      contents: write" in workflow


def test_trusted_provider_ci_wires_full_secret_policy_and_release_dependency():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    integration_job = workflow.partition("\n  integration-test:\n")[2].partition(
        "\n  # ──────────────────────────────────────────────────────────────────"
    )[0]
    release_preflight = workflow.partition("\n  release-preflight:\n")[2].partition(
        "\n  build:\n"
    )[0]

    assert integration_job
    assert "github.event_name == 'pull_request'" not in integration_job
    assert "github.event_name == 'workflow_dispatch'" in integration_job
    for ref in (
        "refs/heads/main",
        "refs/heads/ouroboros",
        "refs/heads/ouroboros-stable",
        "refs/tags/v",
    ):
        assert ref in integration_job

    for secret in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MINIMAX_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
    ):
        assert f"{secret}: ${{{{ secrets.{secret} }}}}" in integration_job

    assert " -rs " in integration_job
    assert "needs: [full-test, integration-test, system-e2e-mock]" in release_preflight


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
