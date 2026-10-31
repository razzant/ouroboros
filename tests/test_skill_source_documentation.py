"""The source migration documents the frozen compatibility surface and retired state honestly."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_hub_route_retirement_names_removed_shapes_and_successors():
    text = (ROOT / "docs/architecture/11-frozen-contracts-v1.md").read_text(encoding="utf-8").split("### 11.4", 1)[1]
    for term in ("/api/marketplace/ouroboroshub/", "catalog", "preview", "install", "update", "uninstall",
                 "publication/clear", "HubCatalogResponse", "HubPreviewResponse", "MarketplaceInstallRequest.adopt",
                 "expected_content_hash", "MarketplaceInstallResponse", "official_hub_verified", "installed",
                 "/api/skill-catalog", "/api/skills/{name}/update", "/api/skills/{name}/uninstall",
                 "/api/skills/{skill}/publication/clear"):
        assert term in text, term
    client = (ROOT / "docs/CLIENT_CONTRACT.md").read_text(encoding="utf-8")
    assert "GET /api/marketplace/ouroboroshub/installed" in client
    assert "11-frozen-contracts-v1.md#114" in client


def test_retired_hub_catalog_row_is_not_a_live_browser_type():
    assert "@typedef {Object} HubCatalogRow" not in (ROOT / "web/modules/api_types.js").read_text(encoding="utf-8")


def test_auto_repair_state_is_documented_as_a_legacy_leftover():
    text = (ROOT / "docs/PERSISTENCE.md").read_text(encoding="utf-8")
    assert "`auto_repair.json` is a legacy leftover; nothing writes it" in text
    owner_state = next(line for line in text.splitlines() if "owner state (`review.json`" in line)
    assert "auto_repair.json" not in owner_state


def test_settings_folder_and_configuration_authority_are_documented():
    text = (ROOT / "docs/architecture/07-configuration.md").read_text(encoding="utf-8")
    assert "never a second authority" in text
    row = next(line for line in text.splitlines() if line.startswith("| OUROBOROS_SKILLS_REPO_PATH |"))
    for fact in ("read in place", "my-folder", "saved source list", "removing that row clears", "Other folders"):
        assert fact in row


@pytest.mark.parametrize("path, facts, stale", [
    ("docs/architecture/13-external-skills-layer.md",
     ("github.com HTTPS without the configured GitHub token", "authentication refusals once with it alone",
      "Other URLs use host credentials", "pre-update record restores an interrupted landing"),
     "Git caches in `state/skill_sources/` use host credentials"),
    ("docs/CREATING_SKILLS.md",
     ("Private github.com HTTPS sources", "this computer's git credentials still apply) is refused for authentication",
      "existing repository, which may be public", "reconnecting keeps your saved name, trust and confirmation"),
     "It is added as a trusted git source"),
    ("docs/CLIENT_CONTRACT.md",
     ("Official (`trusted`) and Community (`review`)", "both unconfirmed",
      "preserves any surviving row's choices"),
     "Official and Community rows with the chosen trust"),
])
def test_source_credentials_and_reconnection_docs_match_the_runtime(path, facts, stale):
    text = " ".join((ROOT / path).read_text(encoding="utf-8").split())
    for fact in facts:
        assert fact in text, fact
    assert stale not in text
