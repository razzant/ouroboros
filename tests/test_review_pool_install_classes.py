"""Every supported install class, end to end: migration M -> review pool -> one wave.

PR-3 (one transition) replaced the review lanes with the catalog's review pool. This
pins, per install class, that the chain still reviews: the lane-era settings document
is read through the product's own read seam (``config.normalize_settings_raw`` ->
``review_pool_migration``), the pool every surface runs is built from the result, and
ONE ``review_change`` wave over that pool on fixture answers (the paid seam stubbed,
the operation real) yields (a) a non-empty pool whose quorum is the lane era's
``adaptive_quorum(len(triad))``, (b) at least one seat asked the ``coupling`` part
(every lane-era install had a scope seat, mandatory in the lane format), and (c) a
ledger record whose rows carry ``parts`` and per-part ``answers``. An install class
with an empty pool or without a coupling seat is a blocker, not a disclosure — the two
controls at the end show the assertions bite.

Classes: the N-1 fixture (6.113.4), the structural copy of Anton's install (contract
§2), the factory OpenRouter install, local-only, compatible-only, one direct provider,
the subscription wizard's preset (a pool document already), and the Colab launches
(the kernel's Drive document: first run with one direct provider, re-run over the N-1
document, the OpenRouter control).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ouroboros.configured_subagents as cs
import ouroboros.review_substrate as substrate
from ouroboros import config as cfg
from ouroboros import review_ledger
from ouroboros import review_pool_migration as m
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.review_ledger import PART_CHANGE, PART_COUPLING, seat_parts
from ouroboros.review_model_routes import adaptive_quorum
from ouroboros.reviewer_slot_config import review_pool_slots
from ouroboros.tools import git as git_mod
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_change import run_review_change
from tests import _contributor_packet_shared as shared
from tests.test_review_pool_migration import N1_DOC, SLOTS, SUBAGENTS, anton_document, api_row, catalog

GOAL = "Make the installed body's helper return the proposal's constant."
SCOPE = "ouroboros/helper.py only; the checklist and tests stay as they are."

# The lane-era document of each install class, exactly as that install's settings.json
# carried it before PR-3 (the N-1 fixture is the recorded one; the others are the
# documents the contract's F6 factory cells and §2 describe).
INSTALL_CLASSES = {
    "nminus1_fixture": lambda: dict(N1_DOC),
    "antons_install": anton_document,
    "factory_openrouter": lambda: {SLOTS: "", "OPENROUTER_API_KEY": "present"},
    "local_only": lambda: {SLOTS: "", "USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "owner/local.gguf",
                           "OUROBOROS_MODEL": "owner/local-main"},
    "compatible_only": lambda: {SLOTS: "", "OPENAI_COMPATIBLE_BASE_URL": "https://llm.example/v1",
                                "OPENAI_COMPATIBLE_API_KEY": "present",
                                "OUROBOROS_MODEL": "openai-compatible::glm-5.3"},
    "direct_openai": lambda: {SLOTS: "", "OPENAI_API_KEY": "present", "OUROBOROS_MODEL": "openai::gpt-5.6-sol"},
}


def _wizard_document() -> dict:
    """The subscription wizard's install-time save: a pool document from the start."""
    from ouroboros.subscription_install_presets import HarnessDiscovery, compile_install_preset
    from tests.test_subscription_install_presets import LIVE_MODELS

    preset = compile_install_preset([HarnessDiscovery(h, LIVE_MODELS[h]) for h in ("codex", "claude", "cursor")])
    assert preset.ok, preset.refusal
    return dict(preset.settings_keys())


def _lane_era_quorum(document: dict) -> int:
    """What the install's commit review required before PR-3: ``adaptive_quorum`` over
    its triad, read by the frozen lane readers (the shipped panel when nothing was
    authored — the factory rows themselves, which carry no scope seat: the pool asks
    every retrieving row the coupling question)."""
    raw = document.get(SLOTS)
    authored = isinstance(raw, str) and bool(raw.strip())
    lanes = m.parse_reviewer_slots(document, raw) if authored else m.factory_lanes(document)
    if authored:
        assert lanes.scope, "every authored lane-era install carried a scope seat (mandatory in the lane format)"
    return adaptive_quorum(len(lanes.triad))


def _coupling_answer(seat_id: str) -> str:
    return json.dumps({"change": [], "change_clean": True,
                       "coupling": shared._coupling_matrix(f"{seat_id} read the checkout")})


def _change_answer(seat_id: str) -> str:
    return json.dumps([{"item": "code_quality", "verdict": "PASS", "severity": "advisory",
                        "reason": f"{seat_id} read the packet"}])


def _answering_substrate(served: list[dict]):
    """``review_substrate.run_review_request`` stand-in (the paid seam only): every seat
    answers the contract its ``parts`` ask — the two-part object for a retrieving seat,
    the change array for a packet seat — and ``served`` records what each seat was."""

    def run_review_request(request, *, slots, drive_root, llm=None, usage_ctx=None):
        reserved = (getattr(usage_ctx, "_review_reserved_operations", None) or {}).get(request.surface) or {}
        actors = []
        for slot in slots:
            parts = seat_parts(slot)
            served.append({"slot_id": slot.slot_id, "model": slot.model, "parts": parts,
                           "route": slot.route, "use_local": slot.use_local})
            if slot.route is ReviewRouteKind.AGENT_SESSION:
                route_id, _, model = str(slot.session_target or slot.model).partition("=")
                usage = {"provider": "claudexor", "delegated_route": route_id, "resolved_model": model,
                         "applied_profile": slot.session_profile, "applied_access": "readonly",
                         "output_conformance": "passed", "verdict_method": "schema"}
            else:
                usage = {"provider": "local" if slot.use_local else "openrouter", "resolved_model": slot.model,
                         "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0}
            actors.append({
                "slot_id": slot.slot_id, "model": slot.model, "status": "ok",
                "raw_text": _coupling_answer(slot.slot_id) if PART_COUPLING in parts else _change_answer(slot.slot_id),
                "usage": usage, "operation_id": str(reserved.get(slot.slot_id) or f"op-{slot.slot_id}"),
                "operation_state": "settled", "late_result_pending": False,
            })
        return SimpleNamespace(actors=actors)

    return run_review_request


@pytest.fixture(autouse=True)
def _migration_state(monkeypatch):
    monkeypatch.setattr(cs, "MAX_CONFIGURED_SUBAGENTS", 26)
    m._MIGRATIONS_SEEN.clear()
    cfg._RETIREMENT_NOTICE_SEEN.clear()
    yield
    m._MIGRATIONS_SEEN.clear()
    cfg._RETIREMENT_NOTICE_SEEN.clear()


@pytest.fixture
def staged_body(tmp_path, monkeypatch):
    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    shared.git(repo, "add", ".gitignore")
    shared.git(repo, "commit", "-q", "-m", "ignore caches")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")  # the proposal touches a protected surface
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    return fixture


def _install(monkeypatch, document: dict) -> dict:
    """Read the document as the product does and make it the applied settings plane."""
    for key in ("OUROBOROS_SUBAGENTS", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_BASE_URL", "USE_LOCAL_MAIN",
                "LOCAL_MODEL_SOURCE", "OUROBOROS_MODEL", "OUROBOROS_REVIEW_MODELS", "GIGACHAT_CREDENTIALS",
                "CLOUDRU_FOUNDATION_MODELS_API_KEY", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    loaded = cfg.normalize_settings_raw(dict(document))
    assert SLOTS not in loaded, "the read seam consumed the lane key"
    for key, value in loaded.items():
        if isinstance(value, bool):
            monkeypatch.setenv(key, "true" if value else "false")
        elif isinstance(value, (str, int, float)):
            monkeypatch.setenv(key, str(value))
    return loaded


def _responded(row: dict) -> set:
    return {part for part, answer in row["answers"].items() if answer["status"] == "responded"}


def _one_wave(tmp_path, monkeypatch, repo: Path):
    served: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", _answering_substrate(served))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
    result = run_review_change(ctx, root="system_repo", surface="change", goal=GOAL, scope=SCOPE, subject="index")
    return result, review_ledger.load_record(ctx.drive_root, result["record_id"]), served


@pytest.mark.parametrize("install", sorted(INSTALL_CLASSES))
def test_each_lane_era_install_migrates_into_a_pool_that_reviews_in_one_wave(install, staged_body, tmp_path,
                                                                             monkeypatch):
    document = INSTALL_CLASSES[install]()
    lane_quorum = _lane_era_quorum(document)
    loaded = _install(monkeypatch, document)
    pool = review_pool_slots()
    # (a) the pool is not empty and its quorum is the lane era's.
    assert pool, f"{install}: the migrated catalog has no marked row"
    assert adaptive_quorum(len(pool)) == lane_quorum, install
    marked = [row["subagent_id"] for row in json.loads(loaded[SUBAGENTS])["items"] if row.get("review_eligible")]
    assert [slot.slot_id for slot in pool] == marked, install

    result, record, served = _one_wave(tmp_path, monkeypatch, Path(staged_body["repo"]))
    assert (result["aggregate"], result["state"]) == ("PASS", "settled"), (install, result)
    rows = record["rows"]
    assert [row["seat_id"] for row in rows] == marked and record["panel"]["seats"] == len(pool), install
    assert record["verdict"]["quorum"] == {
        "required": lane_quorum, "responded": len(pool), "assigned": len(pool),
        "parts": record["verdict"]["quorum"]["parts"]}, install
    # (b) at least one seat read the subject itself and was asked the coupling part.
    coupling_seats = [row for row in rows if PART_COUPLING in row["parts"]]
    assert coupling_seats, f"{install}: no seat was asked the coupling part"
    assert record["verdict"]["per_question"] == {PART_CHANGE: "PASS", PART_COUPLING: "PASS"}, install
    # (c) the record's rows carry ``parts`` and per-part ``answers``, as served: the
    # asked parts answered, the part a packet seat was not asked marked so.
    assert {row["seat_id"]: tuple(row["parts"]) for row in rows} == {s["slot_id"]: s["parts"] for s in served}
    for row in rows:
        assert _responded(row) == set(row["parts"]), (install, row["seat_id"], row["answers"])
        assert all(a["status"] == "not_asked" for p, a in row["answers"].items() if p not in row["parts"]), install
    assert all(row["answers"][PART_COUPLING]["coverage"] != "missing" for row in coupling_seats), install


def test_the_subscription_wizards_preset_is_a_pool_that_reviews_in_one_wave(staged_body, tmp_path, monkeypatch):
    document = _wizard_document()
    loaded = _install(monkeypatch, {**document, "OPENROUTER_API_KEY": "present"})
    pool = review_pool_slots()
    assert pool and adaptive_quorum(len(pool)) == adaptive_quorum(3) == 2  # the lane era's three-seat panel
    assert all(slot.route is ReviewRouteKind.AGENT_SESSION for slot in pool), "the wizard mints session reviewers"
    marked = [row["subagent_id"] for row in json.loads(loaded[SUBAGENTS])["items"] if row.get("review_eligible")]
    result, record, _served = _one_wave(tmp_path, monkeypatch, Path(staged_body["repo"]))
    assert (result["aggregate"], result["state"]) == ("PASS", "settled"), result
    assert [row["seat_id"] for row in record["rows"]] == marked
    assert record["verdict"]["quorum"]["required"] == 2
    assert all(PART_COUPLING in row["parts"] and _responded(row) == {PART_CHANGE, PART_COUPLING}
               for row in record["rows"])
    assert record["verdict"]["per_question"][PART_COUPLING] == "PASS"


# Colab: the kernel builds the Drive document from the collected secrets and writes it
# (``build_colab_settings`` -> ``write_colab_settings``), the server reads it back. The
# pool that write pins must run on the provider the install holds a credential for
# (VD3-02): on the first run (the seed is the channel alone) and on a re-run over a
# lane-era Drive document alike. The OpenRouter launch is the control: the default
# aggregator keeps its shipped rows.
COLAB_LAUNCHES = {
    "colab_first_run_openai": ({"OPENAI_API_KEY": "present"}, {}),
    "colab_rerun_over_nminus1_openai": ({"OPENAI_API_KEY": "present"}, N1_DOC),
    "colab_first_run_openrouter": ({"OPENROUTER_API_KEY": "present"}, {}),
}


@pytest.mark.parametrize("launch", sorted(COLAB_LAUNCHES))
def test_a_colab_launch_pins_the_pool_its_own_provider_can_run(launch, staged_body, tmp_path, monkeypatch):
    from ouroboros.colab_bootstrap import build_colab_settings, write_colab_settings
    from ouroboros.provider_models import model_has_credentials_in_settings
    from ouroboros.subscription_install_presets import factory_review_rows

    secret, drive_document = COLAB_LAUNCHES[launch]
    seed = {**drive_document, "OUROBOROS_UPDATE_CHANNEL": "stable"}  # the notebook's seed
    settings = build_colab_settings({"TELEGRAM_BOT_TOKEN": "present", **secret}, runtime_mode="pro",
                                    existing=seed, drive_document_present=bool(drive_document))
    path = write_colab_settings(tmp_path / "colab_drive", settings)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert SLOTS not in document and SUBAGENTS in document, "the Drive document is a pool document"
    loaded = _install(monkeypatch, document)  # the server's read of the Drive document
    pool = review_pool_slots()
    assert pool and adaptive_quorum(len(pool)) == _lane_era_quorum({SLOTS: "", **secret}), launch
    # The rows are the shipped panel of the provider whose key the document holds — every
    # one runnable with the install's own credentials, none pinned for a provider it lacks.
    assert [slot.model for slot in pool] == [row["route"]["target_id"] for row in factory_review_rows(secret)], launch
    assert all(model_has_credentials_in_settings(slot.model, loaded) for slot in pool), [s.model for s in pool]
    result, record, _served = _one_wave(tmp_path, monkeypatch, Path(staged_body["repo"]))
    assert (result["aggregate"], result["state"]) == ("PASS", "settled"), (launch, result)
    assert [row["seat_id"] for row in record["rows"]] == [slot.slot_id for slot in pool]
    assert record["verdict"]["per_question"] == {PART_CHANGE: "PASS", PART_COUPLING: "PASS"}, launch


# --- the controls: the assertions above bite -------------------------------------------


def test_a_catalog_without_a_review_mark_is_an_empty_pool_and_no_wave_is_paid(staged_body, tmp_path, monkeypatch):
    """A pool document (no lane key) whose rows carry no mark: the migration has
    nothing to do, the pool is empty, and the wave refuses deterministically before
    any reviewer is paid — never a default panel."""
    from ouroboros.tools.review_change import ReviewChangeArgumentError

    _install(monkeypatch, {SUBAGENTS: catalog(api_row("scout", "openai/gpt-5.6-luna")), "OPENROUTER_API_KEY": "present"})
    assert review_pool_slots() == []
    served: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", _answering_substrate(served))
    ctx = ToolContext(repo_dir=Path(staged_body["repo"]), drive_root=tmp_path / "drive")
    with pytest.raises(ReviewChangeArgumentError, match="no seat for the change question"):
        run_review_change(ctx, root="system_repo", surface="change", goal=GOAL, scope=SCOPE, subject="index")
    assert served == [] and not list((tmp_path / "drive").rglob("*.json"))


def test_a_pool_of_packet_seats_only_leaves_the_coupling_question_not_performed(staged_body, tmp_path, monkeypatch):
    """Every seat packet: nobody is asked ``coupling``, so the wave settles
    NOT_PERFORMED — the (b) assertion is a real property of the pool, not of the stub."""
    _install(monkeypatch, {SUBAGENTS: catalog(
        api_row("p1", "openai/gpt-5.6-sol", "high", review_eligible=True, delivery="packet"),
        api_row("p2", "openai/gpt-5.6-terra", "high", review_eligible=True, delivery="packet"),
        api_row("p3", "google/gemini-3.8-flash", "high", review_eligible=True, delivery="packet"),
    ), "OPENROUTER_API_KEY": "present"})
    result, record, served = _one_wave(tmp_path, monkeypatch, Path(staged_body["repo"]))
    assert [s["parts"] for s in served] == [(PART_CHANGE,)] * 3
    assert result["aggregate"] == "NOT_PERFORMED" and record["verdict"]["reason"] == "coupling_not_performed"
    assert record["verdict"]["per_question"][PART_COUPLING] == "not_performed"
    assert all(_responded(row) == {PART_CHANGE} and row["answers"][PART_COUPLING]["status"] == "not_asked"
               for row in record["rows"])
