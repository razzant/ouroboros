"""Imported by a test module whose cases send through a DeepSeek route: its autouse fixture then applies there alone.

Such a send pins its tariff first (``pricing_catalog.pin_tariff``), which may fetch the public price catalog and read
the selected key's balance. A case that does not script those reads gets a refusal instead, so no fake key reaches
DeepSeek and no recorded transport answers a read it never scripted; the price stays unknown, as without a catalog.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_unscripted_tariff_reads(monkeypatch):
    from ouroboros import pricing_catalog

    def refuse(*_args, **_kwargs):
        raise RuntimeError("this case does not script a tariff read")

    monkeypatch.setattr(pricing_catalog, "_fetch_catalog", refuse)
    monkeypatch.setattr(pricing_catalog, "_fetch_currency", refuse)
    monkeypatch.setattr(pricing_catalog, "_currency", {})
    monkeypatch.setattr(pricing_catalog, "_currency_fetching", set())
