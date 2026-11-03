"""Lane M's unit tests consume the host wake-cause seam independently of Lane S."""
import pytest


@pytest.fixture(autouse=True)
def wake_cause_seam(monkeypatch):
    from ouroboros import consciousness_authority

    monkeypatch.setattr(consciousness_authority, "wake_cause",
                        lambda metadata: str(metadata.get("wake_cause") or ""))
