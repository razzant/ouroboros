"""Route keys read the settings document as it is now (``route_choice``), in a task or not.

A reused worker mirrors every task's effective values into its own environment at each task start
(``subagent_runtime.apply_task_start_settings``), so a key the owner cleared since would come back
from that mirror if the live view fell back to the environment. It never does: the view is the
effective document over ``skill_environ`` (this tree's own mirrored values dropped by digest, an
operator's own variable kept), memoised behind the document's stat; an unreadable document makes
credential keys unavailable rather than stale. Every other setting keeps the task's entry snapshot, and
so do the route keys of a process with no saved document at all (an env-configured one). A live read of a
document the owner's settings writer recorded also names the saved API connection it dispatches on, never by its key
(``provider_connections``); any other read names none.
"""

from __future__ import annotations

import json
import os

import pytest

from ouroboros import config, subagent_runtime
from tests.test_task_settings_snapshot import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated, sealed root


def _document(label, **extra):
    return {
        "OUROBOROS_MODEL": f"openai::{label}", "OUROBOROS_MODEL_LIGHT": f"openai::{label}-light",
        "OUROBOROS_MODEL_FALLBACKS": "", "OPENAI_API_KEY": f"key-{label}", "OPENROUTER_API_KEY": f"router-{label}",
        "OPENAI_COMPATIBLE_BASE_URL": f"https://{label}.invalid/v1", "OPENAI_COMPATIBLE_API_KEY": f"compat-{label}",
        "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": f"account-{label}"}),
        "OUROBOROS_EFFORT_TASK": "high" if label == "old" else "low", **extra,
    }


def _targets():
    from ouroboros.llm import LLMClient

    client = LLMClient()
    return {name: client._resolve_remote_target(f"{name}::probe") for name in ("openrouter", "openai", "openai-compatible")}


def test_reused_worker_cannot_resurrect_a_cleared_or_disabled_key(monkeypatch):
    """r2_opus §(3): the mirrored environment of a reused worker never serves a key the owner cleared."""
    save_owner(_document("old"))
    old_task = subagent_runtime.apply_task_start_settings()  # the worker's environment now mirrors every key
    assert os.environ["OPENROUTER_API_KEY"] == "router-old" and os.environ["OPENAI_COMPATIBLE_API_KEY"] == "compat-old"
    save_owner(_document("old", OPENROUTER_API_KEY="", OPENAI_API_KEY="", OPENAI_COMPATIBLE_API_KEY="",
                         OPENAI_COMPATIBLE_BASE_URL=""))
    # No new task started: the old task's bound snapshot and the worker's environment still hold the keys.
    with config.task_settings_scope(old_task):
        assert old_task.environ["OPENROUTER_API_KEY"] == "router-old"
        assert config.runtime_setting("OPENROUTER_API_KEY") is None and config.runtime_setting("OPENAI_API_KEY") is None
        targets = _targets()
        assert targets["openrouter"]["api_key"] == "" and targets["openai"]["api_key"] == ""
        compatible = targets["openai-compatible"]
        assert compatible["api_key"] == "" and compatible["base_url"] == "" and compatible["keyless"] is False
        assert "OPENROUTER_API_KEY" not in config.runtime_environ()  # a child (vision) gets no stale credential either
        assert config.runtime_settings()["OPENROUTER_API_KEY"] == ""
    assert os.environ["OPENROUTER_API_KEY"] == "router-old"  # the mirror itself is untouched: it is simply never read


def test_an_added_key_replaced_disabled_or_removed_mid_task_is_read_at_the_next_call(monkeypatch):
    """The named half of r2_opus §(3): within one task a replaced row key serves the next call, and a row
    the owner turns off or removes refuses there; the built-in key never stands in for it."""
    from ouroboros.llm_attempt import ConnectionUnavailable

    row = {"kind": "openrouter", "name": "credits", "api_key": "row-old", "enabled": True}
    route = "openrouter@credits::anthropic/claude-opus-5.5"
    save_owner({**_document("old"), "OUROBOROS_CONNECTIONS": [row]})
    task = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(task):
        from ouroboros.llm import LLMClient

        assert LLMClient()._resolve_remote_target(route)["api_key"] == "row-old"
        save_owner({**_document("old"), "OUROBOROS_CONNECTIONS": [{**row, "api_key": "row-new"}]})
        assert LLMClient()._resolve_remote_target(route)["api_key"] == "row-new"
        for saved, reason in (([{**row, "enabled": False}], "disabled"), ([], "missing")):
            save_owner({**_document("old"), "OUROBOROS_CONNECTIONS": saved})
            with pytest.raises(ConnectionUnavailable) as refused:
                LLMClient()._resolve_remote_target(route)
            assert refused.value.reason == reason and "router-old" not in str(refused.value)


def test_without_a_saved_document_route_keys_read_as_every_other_key(monkeypatch):
    """An operator-run process (a bench harness, an env-configured container) has nothing to resurrect:
    its environment is the configuration, read as before, entry snapshot included."""
    from ouroboros import route_choice

    assert not config.SETTINGS_PATH.exists()
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::env-main")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    assert route_choice.route_settings_view() is route_choice._ABSENT
    assert config.runtime_setting("OUROBOROS_MODEL") == "openai::env-main" and config.runtime_setting("OPENAI_API_KEY") == "env-key"
    assert config.runtime_setting("OUROBOROS_MODEL_FALLBACKS", "") == ""  # no proposal stands in for an absent chain
    task = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(task):
        monkeypatch.setenv("OUROBOROS_MODEL", "openai::later")
        assert config.runtime_setting("OUROBOROS_MODEL") == "openai::env-main"  # the entry snapshot, as before


def test_a_genuine_operator_variable_still_serves_the_built_in(monkeypatch):
    """An operator's own variable (Docker ``-e``) stands in for a key the document never authored."""
    save_owner({"OUROBOROS_MODEL": "openai::main", "OPENAI_API_KEY": "key-doc"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "operator-key")
    task = subagent_runtime.apply_task_start_settings()
    save_owner({"OUROBOROS_MODEL": "openai::main", "OPENAI_API_KEY": ""})  # the owner clears the authored key only
    with config.task_settings_scope(task):
        assert config.runtime_setting("OPENROUTER_API_KEY") == "operator-key"
        assert _targets()["openrouter"]["api_key"] == "operator-key"
        assert config.runtime_setting("OPENAI_API_KEY") is None  # cleared: the mirror of key-doc never answers


def test_two_rounds_of_one_worker_read_the_saved_model_and_key_between_them(monkeypatch):
    """r2_fable's runner-up: a save between two rounds of one task reaches round 2; round 1's own row stays."""
    from ouroboros import route_choice
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option

    save_owner(_document("old"))
    task = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(task):
        round_one = (config.runtime_setting("OUROBOROS_MODEL"), _targets()["openai"]["api_key"],
                     model_role_option(MODEL_ACCOUNTS_KEY, "main"), route_choice.following_route("main"))
        assert round_one == ("openai::old", "key-old", "account-old", ("openai::old", False))
        save_owner(_document("new"))
        round_two = (config.runtime_setting("OUROBOROS_MODEL"), _targets()["openai"]["api_key"],
                     model_role_option(MODEL_ACCOUNTS_KEY, "main"), route_choice.following_route("main"))
        assert round_two == ("openai::new", "key-new", "account-new", ("openai::new", False))
        assert config.get_light_model() == "openai::new-light"
        # The task's entry snapshot is untouched and still serves everything that is not a route key.
        assert task.environ["OUROBOROS_MODEL"] == "openai::old" and config.runtime_setting("OUROBOROS_EFFORT_TASK") == "high"
        assert config.runtime_settings()["OUROBOROS_EFFORT_TASK"] == "high"
        assert config.resolve_effort("task") == "high"


def test_an_unreadable_document_makes_credentials_unavailable_never_stale(monkeypatch):
    save_owner(_document("old"))
    task = subagent_runtime.apply_task_start_settings()
    config.SETTINGS_PATH.write_text("{broken", encoding="utf-8")
    with config.task_settings_scope(task):
        assert config.runtime_setting("OPENAI_API_KEY") is None and _targets()["openai"]["api_key"] == ""
        assert "OPENAI_API_KEY" not in config.runtime_settings() and "OPENAI_API_KEY" not in config.runtime_environ()
        assert config.runtime_setting("OUROBOROS_MODEL") == "openai::old"  # a model name is not a credential: the entry view
        config.SETTINGS_PATH.unlink()  # repaired by moving it aside: a save never writes over an unreadable document
        save_owner(_document("new"))  # the failure is never memoised: the repaired document answers at once
        assert config.runtime_setting("OPENAI_API_KEY") == "key-new" and config.runtime_setting("OUROBOROS_MODEL") == "openai::new"


def test_structured_route_keys_keep_their_shapes_in_each_view(monkeypatch):
    """A saved connection list is a list in the settings view and JSON text in the env-shaped read;
    it is never exported to a child environment (``SETTINGS_KEYS_NOT_EXPORTED_TO_ENV`` semantics)."""
    rows = [{"kind": "openrouter", "name": "credits", "api_key": "secret-row", "enabled": True}]
    save_owner({**_document("old"), "OUROBOROS_CONNECTIONS": rows})
    task = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(task):
        assert config.runtime_settings()["OUROBOROS_CONNECTIONS"] == rows
        assert json.loads(config.runtime_setting("OUROBOROS_CONNECTIONS")) == rows
        assert config.runtime_setting("OUROBOROS_MODEL_ACCOUNTS") == json.dumps({"main": "account-old"}, separators=(",", ":"))
        assert "OUROBOROS_CONNECTIONS" not in config.runtime_environ()
        view = config.runtime_settings()
        view["OUROBOROS_CONNECTIONS"][0]["api_key"] = "mutated"  # a private copy, never the memo
        assert config.runtime_settings()["OUROBOROS_CONNECTIONS"] == rows


def test_the_live_view_is_the_same_object_until_the_document_or_operator_variables_change(monkeypatch):
    from ouroboros import route_choice

    save_owner(_document("old"))
    route_choice.route_settings_view()  # the loader's first read may persist a compat migration: one more stamp
    first = route_choice.route_settings_view()
    assert first is not None and route_choice.route_settings_view()[0] is first[0]
    monkeypatch.setenv("OPENROUTER_API_KEY", "operator-key")  # an operator variable is part of the key
    second = route_choice.route_settings_view()
    assert second[0] is not first[0] and second[1]["OPENROUTER_API_KEY"] == "router-old"  # the saved key still wins
    save_owner(_document("new"))
    assert route_choice.route_settings_view()[1]["OPENROUTER_API_KEY"] == "router-new"


def test_the_web_search_source_moves_with_its_model_within_a_task(monkeypatch):
    """Source and Model are one choice: a save during a task reaches its next search together, never the
    new Model under the Source the task started with."""
    from ouroboros.search_routes import resolve_web_search_route

    search = {"ANTHROPIC_API_KEY": "anthropic-key"}
    save_owner(_document("old", OUROBOROS_WEBSEARCH_BACKEND="anthropic", OUROBOROS_WEBSEARCH_MODEL="anthropic::old-search", **search))
    task = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(task):
        first = resolve_web_search_route()
        assert first["source"] == "anthropic"
        assert [leg["model"] for leg in first["legs"] + first["unavailable"]] == ["old-search"]
        save_owner(_document("old", OUROBOROS_WEBSEARCH_BACKEND="openai", OUROBOROS_WEBSEARCH_MODEL="gpt-new", **search))
        second = resolve_web_search_route()
        assert second["source"] == "openai" and "error" not in second and "ignored_legacy_model" not in second
        assert [(leg["source"], leg["model"]) for leg in second["legs"]] == [("openai", "gpt-new")]
        assert task.environ["OUROBOROS_WEBSEARCH_BACKEND"] == "anthropic"  # the entry snapshot itself is untouched


_ROUTER, _CREDITS = "anthropic/claude-opus-5.5", "openrouter@credits::anthropic/claude-opus-5.5"


def _account(route=_ROUTER):
    from ouroboros.llm import LLMClient

    return LLMClient()._resolve_remote_target(route).get("dispatch_account", "")


def test_a_live_route_names_the_saved_connection_it_dispatches_on(monkeypatch):
    """A later answer may wake its room only on the connection its call ran on (``extension_late_result``): a live
    read names that saved connection by the random revision the owner's settings writer recorded, never by its
    credential, the same through an unrelated save, a restart or another process, and anew only where that
    connection's own saved fields change."""
    from types import SimpleNamespace

    from ouroboros import provider_connections, route_choice
    from ouroboros.extension_late_result import _captured_route

    row = {"kind": "openrouter", "name": "credits", "api_key": "row-old"}
    unchanged = {**_document("old"), "OUROBOROS_CONNECTIONS": [row]}
    save_owner(unchanged)
    built_in, named = _account(), _account(_CREDITS)
    assert built_in.startswith("api:openrouter#") and named.startswith("api:openrouter@credits#")
    record = (config.DATA_DIR / "state" / "connection_revisions.json").read_text(encoding="utf-8")
    assert not any(secret in text for secret in ("router-old", "row-old", "key-old", "compat-old")
                   for text in (built_in, named, record))
    call = {"model": _ROUTER, "physical_provider": "openrouter", "physical_model": _ROUTER,
            "dispatch_account": built_in, "use_local": False}
    assert _captured_route(SimpleNamespace(_current_llm_call_meta=call))["dispatch_account"] == built_in

    save_owner({**unchanged, "OUROBOROS_TOOL_TIMEOUT_SEC": 601})  # an unrelated save
    assert (_account(), _account(_CREDITS)) == (built_in, named)
    monkeypatch.setattr(provider_connections, "_RECORD", ())  # a restart, or another process, reads the same files
    monkeypatch.setattr(route_choice, "_VIEW", None)
    assert (_account(), _account(_CREDITS)) == (built_in, named)

    save_owner({**unchanged, "OUROBOROS_TOOL_TIMEOUT_SEC": 601, "OPENROUTER_API_KEY": "router-replaced"})
    assert _account() not in ("", built_in) and _account(_CREDITS) == named  # only the replaced key's connection moves
    save_owner({**unchanged, "OUROBOROS_CONNECTIONS": [{**row, "api_key": "row-new"}]})
    assert _account(_CREDITS) not in ("", named)


def test_a_document_the_owner_writer_did_not_record_names_no_connection(monkeypatch):
    """A document the settings writer's record does not name (one saved before the first recorded save, an edit by
    hand, a writer that keeps no record) names no connection, so no earlier call's connection can match it; the next
    owner save gives a new random revision instead of carrying one from that document."""
    config.SETTINGS_PATH.write_text(json.dumps(_document("old"), indent=2), encoding="utf-8")  # an older install's
    assert _account() == ""
    save_owner(_document("old"))
    before = _account()
    assert before.startswith("api:openrouter#")
    on_disk = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
    config.SETTINGS_PATH.write_text(json.dumps({**on_disk, "OUROBOROS_TOOL_TIMEOUT_SEC": 601}, indent=2),
                                    encoding="utf-8")  # an edit by hand, even one that leaves the connection alone
    assert _account() == ""
    save_owner({**_document("old"), "OUROBOROS_TOOL_TIMEOUT_SEC": 601})
    after = _account()
    assert after.startswith("api:openrouter#") and after != before


def test_no_connection_is_named_beside_an_operator_variable_override_or_unsaved_document(monkeypatch):
    """Where the call's credential may not be the saved one (an operator's variable, a one-off key or mapping, no
    saved document at all), nothing names a connection: such a call's later answer stays readable by its letter id."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_routing import resolve_target

    monkeypatch.setenv("OPENROUTER_API_KEY", "operator-key")
    assert _account() == ""  # no saved document: the environment is the configuration
    save_owner(_document("old"))
    target = LLMClient()._resolve_remote_target(_ROUTER)
    assert target["api_key"] == "router-old" and "dispatch_account" not in target  # the variable could stand in
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert _account().startswith("api:openrouter#")
    assert "dispatch_account" not in resolve_target(_ROUTER, {"OPENROUTER_API_KEY": "mapping-key"})
    assert "dispatch_account" not in LLMClient(api_key="one-off")._resolve_remote_target(_ROUTER)
