"""The Settings › Models (Context) passive Nano 80k fact: one real consumer path.

``GET /api/settings`` carries ``_meta.nano_publication`` from
``nano_publication.read_root_sample`` — one small dated sample read on the
canonical drive, with the effective Main provider. The past count and the
scoped check on new memory are separate facts. Synthetic data only.
"""
import json

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import ouroboros.local_model as local_model
from ouroboros.gateway import settings as settings_mod
from ouroboros.gateway.schema import contract_document
from ouroboros.nano_publication import QUALIFIED_WINDOW

NANO_SETTINGS = ({"OUROBOROS_CONTEXT_MODE": "nano"}, {})


def _client(tmp_path, monkeypatch, settings=NANO_SETTINGS, provider=None, evidence=None):
    from ouroboros import nano_host_publication

    monkeypatch.setattr(nano_host_publication, "current_rules_digest", lambda: "rules-fixture")
    monkeypatch.setattr(settings_mod, "settings_read_view", lambda **_: settings)
    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (dict(s), False, []))
    if provider is not None:
        monkeypatch.setattr(settings_mod, "_active_main_route",
                            lambda *a, **k: {"provider": provider, "use_local": provider == "local"})
    if evidence is not None:
        class _Manager:
            def serving_context_evidence(self, *, bind_endpoint=False):
                return evidence
        monkeypatch.setattr(local_model, "get_manager", _Manager)
    app = Starlette(routes=[Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"])])
    app.state.drive_root = tmp_path
    return TestClient(app)


def _write_sample(root, sample):
    root.joinpath("state").mkdir(parents=True, exist_ok=True)
    root.joinpath("state", "nano_minimum_latest.json").write_text(json.dumps(sample), encoding="utf-8")


EXACT_SAMPLE = {"status": "exact", "scope": "cold_start", "observed_at": "2026-11-26T03:23:46+00:00",
                "context_window": QUALIFIED_WINDOW, "memory_tokens": 22_404, "reply_floor_tokens": 8_000,
                "minimum_reply_allowance_tokens": 8_000,
                "minimum_work_headroom_at_reply_floor_tokens": 13_944,
                "minimum_work_headroom_at_wire_allowance_tokens": 13_944,
                "minimum_input_tokens": 58_056, "sent_max_tokens": 10_000, "actual_input_tokens": 60_000,
                "actual_work_headroom_at_wire_allowance_tokens": 10_000, "_process_id": 4242,
                "_port": 8767, "_rules_sha256": "rules-fixture"}


def test_a_non_local_main_reports_unsupported_route_without_reading_the_drive(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, provider="openai")
    fact = client.get("/api/settings").json()["_meta"]["nano_publication"]
    assert fact == {"status": "not_claimed", "reason": "unsupported_route", "measurement": "unavailable",
                    "current_sample": "unavailable", "pressure": None}


def test_a_local_main_without_a_sample_says_no_exact_count_yet(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, provider="local")
    fact = client.get("/api/settings").json()["_meta"]["nano_publication"]
    assert fact["reason"] == "exact_counter_unproven" and fact["measurement"] == "unavailable"
    assert fact["status"] == "not_claimed" and fact["current_sample"] == "unavailable"


def test_an_unknown_sample_keeps_its_scope_and_date_but_no_numbers(tmp_path, monkeypatch):
    _write_sample(tmp_path, {"status": "unknown", "scope": "cold_start", "observed_at": "2026-11-25T22:00:00Z"})
    client = _client(tmp_path, monkeypatch, provider="local")
    fact = client.get("/api/settings").json()["_meta"]["nano_publication"]
    assert fact["reason"] == "exact_counter_unproven" and fact["scope"] == "cold_start"
    assert fact["observed_at"] == "2026-11-25T22:00:00Z" and "memory_tokens" not in fact


def test_the_serving_bindings_exact_sample_prices_memory_and_overflow(tmp_path, monkeypatch):
    _write_sample(tmp_path, dict(EXACT_SAMPLE))
    evidence = {"confirmed": True, "process_id": 4242, "context_window": QUALIFIED_WINDOW, "port": 8767}
    client = _client(tmp_path, monkeypatch, provider="local", evidence=evidence)
    fact = client.get("/api/settings").json()["_meta"]["nano_publication"]
    # Memory over its 20,000 ceiling is outside; headroom below 15,000 beside the sample's own work text is no
    # shortfall, and the input fits. It still never claims protection.
    assert fact["current_sample"] == "outside_limits" and fact["status"] == "not_claimed"
    assert fact["pressure"] == {"memory_excess_tokens": 2_404, "prepared_overflow_tokens": 0}
    assert fact["measurement"] == "exact" and fact["reason"] == "publication_scope_unqualified"
    assert fact["memory_tokens"] == 22_404 and fact["minimum_input_tokens"] == 58_056
    assert fact["minimum_work_headroom_at_reply_floor_tokens"] == 80_000 - 58_056 - 8_000
    assert fact["actual_work_headroom_at_wire_allowance_tokens"] == 80_000 - 60_000 - 10_000
    assert fact["scope"] == "cold_start" and fact["observed_at"] == EXACT_SAMPLE["observed_at"]
    assert fact["observation"] == "prepared"


def test_work_text_already_in_the_sample_leaves_the_work_floor_unconfirmed_not_short(tmp_path, monkeypatch):
    # Background 34,000 + memory 19,000 + the work text Main already carries 15,000 = 68,000, reply floor 8,192.
    sample = {**EXACT_SAMPLE, "memory_tokens": 19_000, "whole_memory_tokens": 19_000, "minimum_input_tokens": 68_000,
              "reply_floor_tokens": 8_192, "minimum_reply_allowance_tokens": 8_192,
              "minimum_work_headroom_at_reply_floor_tokens": 3_808, "minimum_work_headroom_at_wire_allowance_tokens": 3_808,
              "sent_max_tokens": 8_192, "actual_input_tokens": 68_000, "actual_work_headroom_at_wire_allowance_tokens": 3_808}
    _write_sample(tmp_path, sample)
    evidence = {"confirmed": True, "process_id": 4242, "context_window": QUALIFIED_WINDOW, "port": 8767}
    fact = _client(tmp_path, monkeypatch, provider="local", evidence=evidence).get("/api/settings").json()["_meta"]["nano_publication"]
    assert fact["current_sample"] == "work_floor_unconfirmed" and fact["status"] == "not_claimed"
    assert fact["pressure"] == {"memory_excess_tokens": 0, "prepared_overflow_tokens": 0}
    assert fact["minimum_work_headroom_at_reply_floor_tokens"] == fact["actual_work_headroom_at_wire_allowance_tokens"] == 3_808
    assert fact["actual_input_tokens"] == fact["minimum_input_tokens"] == 68_000 and fact["sent_max_tokens"] == 8_192


def test_a_sample_before_an_owner_rules_change_is_not_numbers(tmp_path, monkeypatch):
    _write_sample(tmp_path, {**EXACT_SAMPLE, "_rules_sha256": "previous-rules"})
    evidence = {"confirmed": True, "process_id": 4242, "context_window": QUALIFIED_WINDOW, "port": 8767}
    fact = _client(tmp_path, monkeypatch, provider="local", evidence=evidence).get("/api/settings").json()["_meta"]["nano_publication"]
    assert fact["reason"] == "counted_rules_changed" and "memory_tokens" not in fact


def test_a_sample_from_another_serving_binding_is_not_numbers(tmp_path, monkeypatch):
    _write_sample(tmp_path, dict(EXACT_SAMPLE))
    client = _client(tmp_path, monkeypatch, provider="local", evidence={"confirmed": False})
    fact = client.get("/api/settings").json()["_meta"]["nano_publication"]
    assert fact["reason"] == "serving_binding_changed" and "memory_tokens" not in fact
    assert fact["status"] == "not_claimed" and fact["scope"] == "cold_start"


def test_a_failing_projection_never_breaks_the_settings_read(tmp_path, monkeypatch):
    import ouroboros.nano_publication as publication
    monkeypatch.setattr(publication, "read_root_sample",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("projection")))
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.json()["_meta"]["nano_publication"] == {
        "status": "not_claimed", "reason": "settings_read_failed", "measurement": "unavailable",
        "current_sample": "unavailable", "pressure": None,
    }


def test_non_nano_omits_the_optional_projection_without_calling_its_reader(tmp_path, monkeypatch):
    import ouroboros.nano_publication as publication
    spy = []
    monkeypatch.setattr(publication, "read_root_sample", lambda *a, **k: spy.append((a, k)))
    client = _client(tmp_path, monkeypatch, settings=({"OUROBOROS_CONTEXT_MODE": "max"}, {}), provider="local")
    assert "nano_publication" not in client.get("/api/settings").json()["_meta"]
    assert spy == []


def test_settings_publishes_the_scoped_fact_independently_of_the_measurement(tmp_path, monkeypatch):
    import ouroboros.nano_publication as publication

    # Producer wire fixture, not imported from the sibling's in-flight checkout.
    fixture = {"status": "protecting", "reason": "checked_controlled_growth", "measurement": "unavailable",
               "current_sample": "unavailable", "pressure": None, "protection_scope": "unchanged_main_background",
               "protection_observed_at": "2026-11-26T04:28:00Z", "qualified_background_tokens": 37_000,
               "qualified_reply_floor_tokens": 8_000}
    calls = []

    def read(root, *, provider):
        calls.append((root, provider))
        return fixture

    monkeypatch.setattr(publication, "read_root_sample", read)
    client = _client(tmp_path, monkeypatch, provider="local")
    assert client.get("/api/settings").json()["_meta"]["nano_publication"] == fixture
    assert calls == [(tmp_path, "local")]

    settings = contract_document()["schemas"]["SettingsReadResponse"]["properties"]["_meta"]
    schema = settings["properties"]["nano_publication"]
    assert "nano_publication" not in settings.get("required", [])
    assert schema["properties"]["status"]["enum"] == ["not_claimed", "protecting"]
    assert schema["properties"]["protection_scope"]["enum"] == ["unchanged_main_background"]
    assert {"protection_observed_at", "qualified_background_tokens", "qualified_reply_floor_tokens"} <= schema["properties"].keys()
