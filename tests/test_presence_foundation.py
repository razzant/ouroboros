"""Focused contracts for the internal skill-defined presence foundation."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from ouroboros.contracts.skill_manifest import SkillManifest
from ouroboros.contracts.skill_payload_policy import (
    PRESENCE_PROFILE_STATE_FILENAME,
    PRESENCE_PROFILE_STATE_STEM,
    SKILL_OWNER_STATE_FILENAMES,
    SKILL_OWNER_STATE_STEMS,
    SKILL_PAYLOAD_CONTROL_FILENAMES,
    is_skill_owner_state_alias,
    is_skill_owner_state_target,
)
from ouroboros.presence_capabilities import (
    PresenceArgumentBinding,
    PresenceResourceTarget,
    PresenceScriptTarget,
    PresenceSelection,
    PresenceState,
    PresenceStateError,
    PresenceToolTarget,
    load_presence_state,
    presence_selection_fingerprint,
    presence_state_fingerprint,
    resolve_presence_profile_state,
    save_presence_state,
)
from ouroboros.presence_profile import (
    PresenceCapabilityRequest,
    PresenceProfile,
    PresenceProfileError,
    parse_presence_profile,
    presence_profile_fingerprint,
    presence_request_fingerprint,
)
from ouroboros.presence_runtime import (
    PresenceRuntimeDefaults,
    PresenceRuntimeError,
    PresenceRuntimeOverrides,
    resolve_presence_runtime,
)
from ouroboros.skill_loader import compute_content_hash
from ouroboros.tools.registry import ToolContext


def _manifest(*, presence=..., skill_type: str = "instruction") -> SkillManifest:
    extras = {} if presence is ... else {"presence": presence}
    return SkillManifest(
        name="community-helper",
        description="Neutral multi-human presence fixture.",
        version="0.1.0",
        type=skill_type,
        raw_extra=extras,
    )


def _request(
    request_id: str,
    *,
    kind: str = "tool",
    required: bool = True,
    operations: tuple[str, ...] = (),
    purpose: str = "",
) -> PresenceCapabilityRequest:
    return PresenceCapabilityRequest(
        request_id=request_id,
        kind=kind,
        required=required,
        operations=operations,
        purpose=purpose,
    )


def _profile(
    *requests: PresenceCapabilityRequest,
    instructions: str = "Participate helpfully in the selected room.",
    topics: tuple[str, ...] = ("community-guidelines",),
    runtime: PresenceRuntimeDefaults | None = None,
) -> PresenceProfile:
    return PresenceProfile(
        schema_version=1,
        instructions=instructions,
        instructions_file=None,
        context_topics=topics,
        runtime_defaults=runtime or PresenceRuntimeDefaults(),
        capability_requests=tuple(requests),
    )


def _write_skill(skill_dir, *, presence_yaml: str | None) -> None:
    skill_dir.mkdir(parents=True)
    presence = "" if presence_yaml is None else f"\npresence:{presence_yaml}"
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: community-helper\n"
        "description: Neutral presence fixture.\n"
        "version: 0.1.0\n"
        "type: instruction"
        f"{presence}\n"
        "---\n"
        "# Community helper\n",
        encoding="utf-8",
    )


def _preflight_ctx(tmp_path) -> ToolContext:
    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    repo_dir.mkdir()
    drive_root.mkdir()
    return ToolContext(repo_dir=repo_dir, drive_root=drive_root)


# ---------------------------------------------------------------------------
# Reviewed profile parsing and fingerprints
# ---------------------------------------------------------------------------


def test_absent_presence_is_exact_noop_and_empty_mapping_uses_defaults(tmp_path):
    assert parse_presence_profile(_manifest(), tmp_path) is None

    profile = parse_presence_profile(_manifest(presence={}), tmp_path)

    assert profile is not None
    assert profile.instructions == ""
    assert profile.context_topics == ()
    assert profile.capability_requests == ()
    assert profile.runtime_defaults == PresenceRuntimeDefaults("main", 10)


def test_instruction_and_extension_profiles_are_allowed_but_script_is_not(tmp_path):
    raw = {"instructions": "Join the discussion when useful."}

    assert parse_presence_profile(_manifest(presence=raw), tmp_path) is not None
    assert parse_presence_profile(_manifest(presence=raw, skill_type="extension"), tmp_path) is not None
    with pytest.raises(PresenceProfileError) as caught:
        parse_presence_profile(_manifest(presence=raw, skill_type="script"), tmp_path)
    assert caught.value.code == "presence_unsupported_skill_type"


@pytest.mark.parametrize("raw", [None, [], "public", 7])
def test_explicit_non_mapping_presence_fails(raw, tmp_path):
    with pytest.raises(PresenceProfileError) as caught:
        parse_presence_profile(_manifest(presence=raw), tmp_path)
    assert caught.value.code == "presence_invalid"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ({"unknown": True}, "presence_unknown_field"),
        ({"schema_version": True}, "presence_unsupported_schema_version"),
        ({"runtime_defaults": None}, "presence_invalid_runtime_defaults"),
        (
            {"runtime_defaults": {"model_slot": "heavy"}},
            "invalid_model_slot",
        ),
        ({"context_topics": None}, "presence_invalid_context_topics"),
        ({"capability_requests": None}, "presence_invalid_capability_requests"),
        ({"instructions": ""}, "presence_invalid_instructions"),
        (
            {"instructions": "inline", "instructions_file": "presence.md"},
            "presence_instructions_conflict",
        ),
    ],
)
def test_profile_shape_is_strict(raw, code, tmp_path):
    with pytest.raises(PresenceProfileError) as caught:
        parse_presence_profile(_manifest(presence=raw), tmp_path)
    assert caught.value.code == code
    assert caught.value.field.startswith("presence")


def test_sidecar_is_confined_utf8_and_part_of_both_hashes(tmp_path):
    skill_dir = tmp_path / "community-helper"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("profile payload\n", encoding="utf-8")
    sidecar = skill_dir / "presence.md"
    sidecar.write_text("First reviewed behavior.\n", encoding="utf-8")
    manifest = _manifest(presence={"instructions_file": "presence.md"})

    first = parse_presence_profile(manifest, skill_dir)
    first_payload_hash = compute_content_hash(skill_dir)
    sidecar.write_text("Second reviewed behavior.\n", encoding="utf-8")
    second = parse_presence_profile(manifest, skill_dir)

    assert first is not None and second is not None
    assert first.instructions_file == "presence.md"
    assert first.instructions == "First reviewed behavior.\n"
    assert presence_profile_fingerprint(first) != presence_profile_fingerprint(second)
    assert first_payload_hash != compute_content_hash(skill_dir)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        "/outside.md",
        "~/outside.md",
        r"folder\presence.md",
        "C:/presence.md",
        "C:presence.md",
        "presence.md:stream",
    ],
)
def test_sidecar_rejects_escaping_or_ambiguous_paths(path, tmp_path):
    with pytest.raises(PresenceProfileError):
        parse_presence_profile(_manifest(presence={"instructions_file": path}), tmp_path)


def test_sidecar_rejects_symlink_escape_and_non_utf8(tmp_path):
    skill_dir = tmp_path / "community-helper"
    skill_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        (skill_dir / "escape.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    with pytest.raises(PresenceProfileError):
        parse_presence_profile(_manifest(presence={"instructions_file": "escape.md"}), skill_dir)

    (skill_dir / "binary.md").write_bytes(b"\xff\xfe")
    with pytest.raises(PresenceProfileError) as caught:
        parse_presence_profile(_manifest(presence={"instructions_file": "binary.md"}), skill_dir)
    assert caught.value.code == "presence_instructions_file_encoding"


def test_sidecar_must_belong_to_the_reviewed_payload_surface(tmp_path):
    skill_dir = tmp_path / "community-helper"
    excluded = skill_dir / ".ouroboros_env"
    excluded.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("reviewed\n", encoding="utf-8")
    (excluded / "presence.md").write_text("unreviewed\n", encoding="utf-8")

    with pytest.raises(PresenceProfileError) as caught:
        parse_presence_profile(
            _manifest(presence={"instructions_file": ".ouroboros_env/presence.md"}),
            skill_dir,
        )
    assert caught.value.code == "presence_instructions_file_unreviewed"


@pytest.mark.parametrize("filename", sorted(SKILL_PAYLOAD_CONTROL_FILENAMES))
def test_sidecar_rejects_skill_lifecycle_control_files(filename, tmp_path):
    skill_dir = tmp_path / "community-helper"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("reviewed\n", encoding="utf-8")
    (skill_dir / filename).write_text("control plane\n", encoding="utf-8")

    with pytest.raises(PresenceProfileError) as caught:
        parse_presence_profile(
            _manifest(presence={"instructions_file": filename}),
            skill_dir,
        )
    assert caught.value.code == "presence_instructions_file_control_plane"


def test_capability_requests_are_strict_and_resource_operations_are_normalized(tmp_path):
    profile = parse_presence_profile(
        _manifest(
            presence={
                "capability_requests": [
                    {
                        "id": "shared-files",
                        "kind": "resource",
                        "purpose": "Read selected shared material.",
                        "required": True,
                        "operations": ["Search", "read", "search"],
                    },
                    {
                        "id": "publish-reply",
                        "kind": "tool",
                        "required": False,
                    },
                ]
            }
        ),
        tmp_path,
    )

    assert profile is not None
    assert profile.capability_requests[0].operations == ("read", "search")
    assert profile.capability_requests[1].operations == ()

    invalid = [
        [{"id": "x", "kind": "tool", "required": "yes"}],
        [{"id": "x", "kind": "resource", "required": True}],
        [{"id": "x", "kind": "tool", "required": True, "operations": ["read"]}],
        [
            {"id": "x", "kind": "tool", "required": True},
            {"id": "x", "kind": "tool", "required": False},
        ],
    ]
    for requests in invalid:
        with pytest.raises(PresenceProfileError):
            parse_presence_profile(_manifest(presence={"capability_requests": requests}), tmp_path)


def test_request_mapping_survives_prompt_topic_runtime_and_purpose_edits():
    original = _request("reply", purpose="Send a useful reply.")
    prose_edit = replace(original, purpose="Publish a useful response.")
    base = _profile(original)
    changed = replace(
        base,
        instructions="Use the latest reviewed style.",
        context_topics=("publishing-guidelines",),
        runtime_defaults=PresenceRuntimeDefaults("light", 6),
        capability_requests=(prose_edit,),
    )

    assert presence_request_fingerprint(original) == presence_request_fingerprint(prose_edit)
    assert presence_profile_fingerprint(base) != presence_profile_fingerprint(changed)
    assert presence_request_fingerprint(original) != presence_profile_fingerprint(base)
    assert len(presence_request_fingerprint(original)) == 64


def test_request_mapping_survives_required_readiness_changes():
    optional = _request("reply", required=False)
    required = replace(optional, required=True)
    selection = PresenceSelection(
        presence_request_fingerprint(optional),
        PresenceToolTarget("builtin", "chat_history"),
    )

    assert presence_request_fingerprint(optional) == presence_request_fingerprint(required)
    resolution = resolve_presence_profile_state(_profile(required), PresenceState((selection,)), global_max_rounds=20)
    assert resolution.active == (selection,)
    assert resolution.missing_required == ()

    optional_resource = _request("shared-files", kind="resource", required=False, operations=("read",))
    required_resource = replace(optional_resource, required=True)
    resource_fingerprint = presence_request_fingerprint(optional_resource)
    resource_selection = PresenceSelection(
        resource_fingerprint,
        PresenceResourceTarget("selected_files", ("read",)),
    )
    publish = _request("publish")
    publish_selection = PresenceSelection(
        presence_request_fingerprint(publish),
        PresenceToolTarget("extension", "publish_reply", "community-actions"),
        (
            PresenceArgumentBinding(
                ("source",),
                "resource",
                resource_request_fingerprint=resource_fingerprint,
            ),
        ),
    )
    dependent = resolve_presence_profile_state(
        _profile(required_resource, publish),
        PresenceState((resource_selection, publish_selection)),
        global_max_rounds=20,
    )
    assert set(dependent.active) == {resource_selection, publish_selection}


# ---------------------------------------------------------------------------
# Runtime defaults and local overrides
# ---------------------------------------------------------------------------


def test_runtime_precedence_cap_and_reset():
    defaults = PresenceRuntimeDefaults("light", 10)
    overridden = resolve_presence_runtime(
        defaults,
        PresenceRuntimeOverrides("main", 14),
        global_max_rounds=12,
    )
    reset = resolve_presence_runtime(defaults, None, global_max_rounds=20)
    builtin = resolve_presence_runtime(None, None, global_max_rounds=200)

    assert overridden.model_slot == "main"
    assert overridden.requested_inline_max_rounds == 14
    assert overridden.inline_max_rounds == 12
    assert overridden.capped is True
    assert reset.model_slot == "light" and reset.inline_max_rounds == 10
    assert builtin.model_slot == "main" and builtin.inline_max_rounds == 10


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PresenceRuntimeDefaults("vision", 10),
        lambda: PresenceRuntimeDefaults("heavy", 10),
        lambda: PresenceRuntimeOverrides("heavy", 10),
        lambda: PresenceRuntimeDefaults("main", True),
        lambda: PresenceRuntimeOverrides(inline_max_rounds=0),
        lambda: resolve_presence_runtime(None, None, global_max_rounds=False),
    ],
)
def test_runtime_values_fail_closed(factory):
    with pytest.raises(PresenceRuntimeError):
        factory()


# ``execution`` (event | call) is additive: parse, override, resolution, and two
# frozen fingerprints that must not move for a profile/state written without it.
# These literals were computed from the code that predates the field.
_PROFILE_FINGERPRINT_BEFORE_EXECUTION = "5ac34b2c70b71616e40d3262212ede130e32396d8019f4418ad1e56fac79e911"
_STATE_FINGERPRINT_BEFORE_EXECUTION = "b537ab97fb003f7c84cdc220df7088b3ca361682856085fcba7034e2daa135e8"


def test_execution_parses_overrides_and_resolves_with_event_as_the_default(tmp_path):
    absent = parse_presence_profile(_manifest(presence={"runtime_defaults": {}}), tmp_path)
    call = parse_presence_profile(
        _manifest(presence={"runtime_defaults": {"execution": "call"}}), tmp_path)
    assert absent.runtime_defaults.execution == "event"
    assert call.runtime_defaults.execution == "call"

    assert resolve_presence_runtime(call.runtime_defaults, None, global_max_rounds=None).execution == "call"
    assert resolve_presence_runtime(
        call.runtime_defaults, PresenceRuntimeOverrides(execution="event"), global_max_rounds=None,
    ).execution == "event"
    assert resolve_presence_runtime(
        absent.runtime_defaults, PresenceRuntimeOverrides(execution="call"), global_max_rounds=None,
    ).execution == "call"
    assert resolve_presence_runtime(None, None, global_max_rounds=None).execution == "event"

    with pytest.raises(PresenceProfileError) as refused:
        parse_presence_profile(_manifest(presence={"runtime_defaults": {"execution": "voice"}}), tmp_path)
    assert refused.value.code == "invalid_execution"
    assert PresenceRuntimeOverrides(execution=None).execution is None  # the override's absence
    for bad in ("", "Call", 1):
        with pytest.raises(PresenceRuntimeError):
            PresenceRuntimeOverrides(execution=bad)
        with pytest.raises(PresenceRuntimeError):
            PresenceRuntimeDefaults(execution=bad)


def test_profile_fingerprint_is_byte_identical_without_execution_and_moves_with_call():
    before = presence_profile_fingerprint(_profile())
    explicit_event = presence_profile_fingerprint(
        _profile(runtime=PresenceRuntimeDefaults(execution="event")))
    with_call = presence_profile_fingerprint(_profile(runtime=PresenceRuntimeDefaults(execution="call")))

    assert before == explicit_event == _PROFILE_FINGERPRINT_BEFORE_EXECUTION
    assert with_call != before


def test_owner_override_door_accepts_execution_and_projects_it_only_when_set():
    from ouroboros.gateway.presence_settings import _parse_overrides, _runtime_projection

    legacy_body = {"expected_state_fingerprint": "f" * 64,
                   "runtime_overrides": {"model_slot": "main", "inline_max_rounds": 7}}
    assert _parse_overrides(legacy_body)[1] == legacy_body["runtime_overrides"]
    call_body = {"expected_state_fingerprint": "f" * 64,
                 "runtime_overrides": {"model_slot": None, "inline_max_rounds": None, "execution": "call"}}
    assert _parse_overrides(call_body)[1]["execution"] == "call"
    with pytest.raises(ValueError):
        _parse_overrides({"expected_state_fingerprint": "f" * 64, "runtime_overrides": {"execution": "call"}})

    projected = _runtime_projection(_profile(), PresenceState())
    assert projected["defaults"] == {"model_slot": "main", "inline_max_rounds": 10}
    assert projected["overrides"] == {"model_slot": None, "inline_max_rounds": None}
    enabled = _runtime_projection(
        _profile(runtime=PresenceRuntimeDefaults(execution="call")),
        PresenceState(runtime_overrides=PresenceRuntimeOverrides(execution="event")),
    )
    assert enabled["defaults"]["execution"] == "call" and enabled["overrides"]["execution"] == "event"


def test_state_fingerprint_and_stored_shape_ignore_an_unset_execution_override(tmp_path):
    assert presence_state_fingerprint(PresenceState()) == _STATE_FINGERPRINT_BEFORE_EXECUTION
    with_call = PresenceState(runtime_overrides=PresenceRuntimeOverrides(execution="call"))
    assert presence_state_fingerprint(with_call) != _STATE_FINGERPRINT_BEFORE_EXECUTION

    # Round trip: an override set to ``call`` is stored and read back; one left unset
    # writes the shape every earlier version wrote (no ``execution`` key at all).
    save_presence_state(tmp_path, "community-helper", with_call,
                        expected_state_fingerprint=_STATE_FINGERPRINT_BEFORE_EXECUTION)
    stored = json.loads((tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME)
                        .read_text(encoding="utf-8"))
    assert stored["runtime_overrides"] == {"model_slot": None, "inline_max_rounds": None, "execution": "call"}
    assert load_presence_state(tmp_path, "community-helper") == with_call
    save_presence_state(tmp_path, "community-helper", PresenceState(),
                        expected_state_fingerprint=presence_state_fingerprint(with_call))
    stored = json.loads((tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME)
                        .read_text(encoding="utf-8"))
    assert stored["runtime_overrides"] == {"model_slot": None, "inline_max_rounds": None}


# ---------------------------------------------------------------------------
# Exact local selections and durable state
# ---------------------------------------------------------------------------


def test_absent_state_does_not_create_a_directory(tmp_path):
    state = load_presence_state(tmp_path, "community-helper")

    assert state == PresenceState()
    assert not (tmp_path / "state").exists()


def test_all_target_and_binding_kinds_roundtrip_through_locked_state(tmp_path):
    resource_request = _request("shared-files", kind="resource", operations=("read", "search"))
    tool_request = _request("publish-reply")
    script_request = _request("format-post", kind="script", required=False)
    resource_fingerprint = presence_request_fingerprint(resource_request)
    bindings = (
        PresenceArgumentBinding(("actor_id",), "actor", ("account_id",)),
        PresenceArgumentBinding(("conversation",), "conversation", ("id",)),
        PresenceArgumentBinding(("message",), "message", ("text",)),
        PresenceArgumentBinding(("destination",), "destination", ("id",)),
        PresenceArgumentBinding(("format",), "static", static_value={"style": "brief"}),
        PresenceArgumentBinding(
            ("source_path",),
            "resource",
            resource_request_fingerprint=resource_fingerprint,
        ),
    )
    selections = (
        PresenceSelection(
            resource_fingerprint,
            PresenceResourceTarget(
                "selected_files",
                ("search", "read"),
                path_prefix="shared",
                bucket="external",
                skill_name="community-helper",
            ),
        ),
        PresenceSelection(
            presence_request_fingerprint(tool_request),
            PresenceToolTarget("extension", "publish_reply", "community-actions"),
            bindings,
        ),
        PresenceSelection(
            presence_request_fingerprint(script_request),
            PresenceScriptTarget("community-helper", "scripts/format.py"),
        ),
    )
    state = PresenceState(
        selections=selections,
        runtime_overrides=PresenceRuntimeOverrides("light", 8),
    )

    saved = save_presence_state(
        tmp_path,
        "community-helper",
        state,
        expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
    )

    assert saved == state
    assert load_presence_state(tmp_path, "community-helper") == state
    assert (tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME).is_file()


def test_state_fingerprints_are_order_independent_but_bind_targets_and_overrides():
    one = PresenceSelection("1" * 64, PresenceToolTarget("builtin", "chat_history"))
    two = PresenceSelection("2" * 64, PresenceToolTarget("mcp", "send", "social"))
    forward = PresenceState((one, two))
    reverse = PresenceState((two, one))

    assert presence_state_fingerprint(forward) == presence_state_fingerprint(reverse)
    assert presence_selection_fingerprint(one) != presence_selection_fingerprint(two)
    assert presence_state_fingerprint(forward) != presence_state_fingerprint(
        PresenceState((one, two), PresenceRuntimeOverrides("light", None))
    )


def test_malformed_state_and_cas_conflict_fail_without_overwrite(tmp_path):
    state_path = tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(PresenceStateError) as malformed:
        load_presence_state(tmp_path, "community-helper")
    assert malformed.value.code == "presence_state_malformed_json"

    state_path.unlink()
    first = PresenceState((PresenceSelection("a" * 64, PresenceToolTarget("builtin", "chat_history")),))
    save_presence_state(
        tmp_path,
        "community-helper",
        first,
        expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
    )
    before = state_path.read_bytes()
    with pytest.raises(PresenceStateError) as conflict:
        save_presence_state(
            tmp_path,
            "community-helper",
            PresenceState(),
            expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
        )
    assert conflict.value.code == "presence_state_conflict"
    assert state_path.read_bytes() == before


@pytest.mark.parametrize(
    ("binding", "expected_code"),
    [
        (
            {
                "argument_path": ["path"],
                "source": "resource",
                "resource_request_fingerprint": int("1" * 64),
            },
            "presence_state_invalid_string",
        ),
        (
            {
                "argument_path": ["path"],
                "source": [],
            },
            "presence_state_invalid_binding_source",
        ),
    ],
)
def test_state_decoder_rejects_non_string_binding_fields(tmp_path, binding, expected_code):
    state_path = tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selections": [
                    {
                        "request_fingerprint": "a" * 64,
                        "target": {
                            "type": "tool",
                            "kind": "builtin",
                            "name": "read_file",
                            "provider": "",
                        },
                        "argument_bindings": [binding],
                    }
                ],
                "runtime_overrides": {
                    "model_slot": None,
                    "inline_max_rounds": None,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PresenceStateError) as malformed:
        load_presence_state(tmp_path, "community-helper")

    assert malformed.value.code == expected_code


def test_persisted_runtime_override_rejects_legacy_heavy(tmp_path):
    state_path = tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selections": [],
                "runtime_overrides": {
                    "model_slot": "heavy",
                    "inline_max_rounds": None,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PresenceStateError) as malformed:
        load_presence_state(tmp_path, "community-helper")

    assert malformed.value.code == "presence_state_invalid_runtime_overrides"


def test_unknown_state_fields_and_noncanonical_skill_names_fail(tmp_path):
    state_path = tmp_path / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selections": [],
                "runtime_overrides": {
                    "model_slot": None,
                    "inline_max_rounds": None,
                },
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PresenceStateError) as unknown:
        load_presence_state(tmp_path, "community-helper")
    assert unknown.value.code == "presence_state_unknown_field"

    with pytest.raises(PresenceStateError):
        load_presence_state(tmp_path, "../community-helper")


def test_target_and_binding_validation_rejects_ambiguous_authority():
    with pytest.raises(PresenceStateError):
        PresenceToolTarget("builtin", "chat_history", "provider")
    with pytest.raises(PresenceStateError):
        PresenceToolTarget("extension", "publish_reply")
    with pytest.raises(PresenceStateError):
        PresenceToolTarget("mcp", "publish_reply")
    with pytest.raises(PresenceStateError):
        PresenceScriptTarget("community-helper", "../format.py")
    with pytest.raises(PresenceStateError):
        PresenceScriptTarget("community-helper", "C:/format.py")
    with pytest.raises(PresenceStateError):
        PresenceScriptTarget("community-helper", "format.py:stream")
    with pytest.raises(PresenceStateError):
        PresenceResourceTarget("selected_files", ("read",), path_prefix="../private")
    with pytest.raises(PresenceStateError):
        PresenceResourceTarget("selected_files", ("read",), path_prefix="C:private")
    with pytest.raises(PresenceStateError):
        PresenceResourceTarget("selected_files", ("read",), path_prefix="shared:stream")
    with pytest.raises(PresenceStateError):
        PresenceArgumentBinding(("value",), "message", static_value="forged")
    with pytest.raises(PresenceStateError):
        PresenceArgumentBinding(("value",), "static", static_value=float("nan"))


def test_argument_bindings_reject_parent_child_overlap():
    with pytest.raises(PresenceStateError) as caught:
        PresenceSelection(
            "a" * 64,
            PresenceToolTarget("builtin", "chat_history"),
            (
                PresenceArgumentBinding(("destination",), "static", static_value={}),
                PresenceArgumentBinding(("destination", "id"), "destination", ("id",)),
            ),
        )
    assert caught.value.code == "presence_state_overlapping_binding"


def test_static_json_is_deeply_immutable_and_fingerprint_stable():
    binding = PresenceArgumentBinding(
        ("options",),
        "static",
        static_value={"labels": ["one", {"nested": True}]},
    )
    selection = PresenceSelection(
        "b" * 64,
        PresenceToolTarget("builtin", "chat_history"),
        (binding,),
    )
    before = presence_selection_fingerprint(selection)

    with pytest.raises(TypeError):
        binding.static_value["extra"] = True
    assert isinstance(binding.static_value["labels"], tuple)
    with pytest.raises(AttributeError):
        binding.static_value["labels"].append("two")
    assert presence_selection_fingerprint(selection) == before


def test_resolution_separates_active_missing_optional_and_orphaned():
    required_tool = _request("reply")
    optional_script = _request("format", kind="script", required=False)
    resource = _request("shared-files", kind="resource", operations=("read", "search"))
    resource_fp = presence_request_fingerprint(resource)
    tool_selection = PresenceSelection(
        presence_request_fingerprint(required_tool),
        PresenceToolTarget("extension", "publish_reply", "community-actions"),
        (
            PresenceArgumentBinding(
                ("source",),
                "resource",
                resource_request_fingerprint=resource_fp,
            ),
        ),
    )
    resource_selection = PresenceSelection(
        resource_fp,
        PresenceResourceTarget("selected_files", ("search", "read")),
    )
    orphan = PresenceSelection(
        "f" * 64,
        PresenceToolTarget("builtin", "list_files"),
    )
    profile = _profile(required_tool, optional_script, resource)
    state = PresenceState((orphan, tool_selection, resource_selection))

    resolution = resolve_presence_profile_state(profile, state, global_max_rounds=200)

    assert {item.request_fingerprint for item in resolution.active} == {
        presence_request_fingerprint(required_tool),
        resource_fp,
    }
    assert resolution.missing_required == ()
    assert resolution.missing_optional == (optional_script,)
    assert resolution.orphaned == (orphan,)
    assert resolution.required_selections_present is True

    prompt_edit = replace(profile, instructions="Use a newly reviewed voice.")
    edited = resolve_presence_profile_state(prompt_edit, state, global_max_rounds=200)
    assert edited.active == resolution.active
    assert edited.selection_fingerprint == resolution.selection_fingerprint
    assert edited.profile_fingerprint != resolution.profile_fingerprint


def test_partial_resource_mapping_does_not_satisfy_required_request():
    request = _request("shared-files", kind="resource", operations=("read", "search"))
    selection = PresenceSelection(
        presence_request_fingerprint(request),
        PresenceResourceTarget("selected_files", ("read",)),
    )

    resolution = resolve_presence_profile_state(_profile(request), PresenceState((selection,)), global_max_rounds=20)

    assert resolution.active == ()
    assert resolution.missing_required == (request,)
    assert resolution.orphaned == (selection,)
    assert resolution.required_selections_present is False


# ---------------------------------------------------------------------------
# Owner-state protection and deterministic preflight integration
# ---------------------------------------------------------------------------


def test_presence_state_filename_and_alias_are_owner_protected(tmp_path):
    assert PRESENCE_PROFILE_STATE_FILENAME in SKILL_OWNER_STATE_FILENAMES
    assert PRESENCE_PROFILE_STATE_STEM in SKILL_OWNER_STATE_STEMS
    data_root = tmp_path / "data"
    state_path = data_root / "state" / "skills" / "community-helper" / PRESENCE_PROFILE_STATE_FILENAME
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    alias = tmp_path / "state-alias.json"
    os.link(state_path, alias)

    assert is_skill_owner_state_target(state_path, data_root) is True
    assert is_skill_owner_state_alias(alias, data_root) is True

def test_skill_preflight_absent_presence_keeps_output_shape(tmp_path, monkeypatch):
    ctx = _preflight_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    _write_skill(skills_root / "community-helper", presence_yaml=None)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))

    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    payload = json.loads(_handle_skill_preflight(ctx, skill="community-helper"))
    assert payload["ok"] is True
    assert "presence" not in payload


def test_skill_preflight_reports_valid_and_invalid_presence(tmp_path, monkeypatch):
    ctx = _preflight_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root / "community-helper",
        presence_yaml=(
            "\n  instructions: Participate when useful.\n"
            "  runtime_defaults:\n"
            "    model_slot: main\n"
            "    inline_max_rounds: 10\n"
        ),
    )
    _write_skill(skills_root / "invalid-helper", presence_yaml=" null")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))

    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    valid = json.loads(_handle_skill_preflight(ctx, skill="community-helper"))
    invalid = json.loads(_handle_skill_preflight(ctx, skill="invalid-helper"))
    assert valid["ok"] is True
    assert valid["presence"][0]["ok"] is True
    assert len(valid["presence"][0]["profile_fingerprint"]) == 64
    assert invalid["ok"] is False
    assert invalid["presence"][0]["code"] == "presence_invalid"


def test_skill_preflight_unexpected_profile_failure_is_fail_closed(tmp_path, monkeypatch):
    ctx = _preflight_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root / "community-helper",
        presence_yaml="\n  instructions: Participate when useful.\n",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))

    import ouroboros.presence_profile as profile_module
    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    def _crash(*_args, **_kwargs):
        raise RuntimeError("synthetic parser crash")

    monkeypatch.setattr(profile_module, "parse_presence_profile", _crash)
    payload = json.loads(_handle_skill_preflight(ctx, skill="community-helper"))

    assert payload["ok"] is False
    assert payload["presence"][0]["code"] == "presence_internal_error"
    assert "synthetic parser crash" not in payload["presence"][0]["detail"]
