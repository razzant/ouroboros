"""Wire validation and per-primitive authority for the task-owned Chrome MCP.

The Chrome server owns a Node executor and an MV3 connection. This module owns
neither: it validates its same-connection state and answers the guard socket.
No result text, previous tab, or model-supplied lease is an authority source.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import types
import uuid
from typing import Any

from ouroboros.tools.tool_result import ToolResult

PROTOCOL = "ouroboros_chrome"
VERSION = 1
STATE_TOOL = "browser_state"
TOOLS = frozenset({STATE_TOOL, "browser_connect", "browser_action", "browser_execute",
                   "browser_handoff", "browser_resume", "browser_disconnect"})
CONTROL = frozenset({STATE_TOOL, "browser_connect", "browser_handoff", "browser_resume",
                     "browser_disconnect"})
READ_OPERATIONS = frozenset({"snapshot", "dom_snapshot", "ax_snapshot", "get_text",
                             "get_attribute", "screenshot", "wait", "read", "observe", "cached_text"})
NAVIGATIONS = frozenset({"navigate", "goto", "open_url"})
INPUT_OPERATIONS = frozenset({"click", "double_click", "fill", "type", "press", "select",
                              "check", "uncheck", "drag", "upload", "submit", "scroll"})
OPERATIONS = READ_OPERATIONS | NAVIGATIONS | INPUT_OPERATIONS
READ_RESOURCES = frozenset({"Stylesheet", "Image", "Media", "Font", "Script", "TextTrack",
                            "XHR", "Fetch", "Prefetch", "Manifest", "Other"})


class ChromeStateError(RuntimeError):
    pass


def _identity(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _generation(value: Any) -> bool:
    return type(value) is int and value >= 0


def parse_state(result: Any, owner: tuple[str, str], *, allow_closed: bool = False) -> dict[str, Any]:
    """One MCP text item containing one complete JSON object, bound to this attempt."""
    parts = getattr(result, "content", None) or []
    if (getattr(result, "isError", False) or len(parts) != 1
            or getattr(parts[0], "type", "") != "text"):
        raise ChromeStateError("BROWSER_STATE_UNKNOWN: expected one JSON text result")
    try:
        state = json.loads(parts[0].text)
    except (ValueError, TypeError) as exc:
        raise ChromeStateError("BROWSER_STATE_UNKNOWN: malformed JSON state") from exc
    return validate_state(state, owner, allow_closed=allow_closed)


def validate_state(state: Any, owner: tuple[str, str], *, allow_closed: bool = False) -> dict[str, Any]:
    if (not isinstance(state, dict) or state.get("protocol") != PROTOCOL
            or type(state.get("version")) is not int or state["version"] != VERSION):
        raise ChromeStateError("BROWSER_STATE_UNKNOWN: wrong Chrome protocol or version")
    claimed = state.get("owner")
    if not isinstance(claimed, dict) or (claimed.get("task_id"), str(claimed.get("attempt"))) != owner:
        raise ChromeStateError("BROWSER_OWNER_CHANGED: Chrome state belongs to another task attempt")
    if not _generation(state.get("executor_generation")):
        raise ChromeStateError("BROWSER_STATE_UNKNOWN: executor generation missing")
    connected = state.get("connected")
    closure = state.get("closure")
    if type(connected) is not bool or not isinstance(closure, dict) or any(
            type(closure.get(field)) is not bool for field in ("revoked", "detached")):
        raise ChromeStateError("BROWSER_STATE_UNKNOWN: connection or closure facts missing")
    lease = state.get("lease")
    if connected:
        if not isinstance(lease, dict) or not all(_identity(lease.get(field)) for field in
                                                  ("lease_id", "tab_id", "document_id", "url")):
            raise ChromeStateError("BROWSER_STATE_UNKNOWN: connected lease is incomplete")
        if not _generation(lease.get("generation")) or type(lease.get("guard_armed")) is not bool:
            raise ChromeStateError("BROWSER_STATE_UNKNOWN: lease generation or guard fact missing")
    elif lease is not None:
        raise ChromeStateError("BROWSER_STATE_UNKNOWN: disconnected state has a lease")
    if closure["revoked"] and not allow_closed:
        raise ChromeStateError("BROWSER_SESSION_REVOKED: Chrome owner revoked the connection")
    if closure["detached"] and not allow_closed:
        raise ChromeStateError("BROWSER_SESSION_DETACHED: Chrome extension is detached")
    return state


def lease_key(state: dict[str, Any]) -> tuple[Any, ...]:
    """Lease authority, excluding the mutable URL observed on the same document."""
    lease = state["lease"]
    identity = (state["owner"]["task_id"], str(state["owner"]["attempt"]),
                state["executor_generation"], state["connected"],
                state["closure"]["revoked"], state["closure"]["detached"])
    return identity if lease is None else (*identity, lease["lease_id"],
                                           lease["generation"], lease["tab_id"], lease["document_id"],
                                           lease["guard_armed"])


def same_document_url_refresh(prior: dict[str, Any], current: dict[str, Any]) -> bool:
    """A URL update may retain one live document lease, but no other identity fact."""
    before, after = prior["lease"], current["lease"]
    return bool(
        before is not None and after is not None and before["url"] != after["url"]
        and lease_key(prior) == lease_key(current)
    )


def invocation_reason(facts: Any, invocation: tuple[str, str, dict[str, Any]] | None,
                      last_step: int) -> str:
    """The host's invocation and ordered callback identity, before any state change."""
    if not isinstance(facts, dict) or invocation is None or facts.get("invocation_id") != invocation[0]:
        return "BROWSER_POLICY_UNAVAILABLE: callback invocation does not match the active call"
    step = facts.get("step_id")
    if type(step) is not int or step <= last_step:
        return "BROWSER_POLICY_UNAVAILABLE: callback step must increase within the invocation"
    return ""


def _policy_inputs(ctx: Any) -> tuple[bool, str]:
    """Actor authority from the context carrier; never copy or scan task history."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.consciousness_authority import effective_runtime_mode, is_observe_origin
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
    from ouroboros.tool_access import active_tool_profile
    from ouroboros.tools.core import is_restricted_subagent_profile

    metadata = getattr(ctx, "task_metadata", None)
    mode = effective_runtime_mode(get_runtime_mode(), metadata)
    # The shared profile helper's Cyber exemption uses the INSTALL mode. An
    # acting task with a stricter cap must retain its resource restrictions.
    restricted = (is_restricted_subagent_profile(ctx) or is_observe_origin(metadata)
                  or (active_tool_profile(ctx) == "acting_subagent"
                      and not mode_has_unrestricted_agency(mode)))
    return restricted, mode


def target_reason(ctx: Any, url: str) -> str:
    from ouroboros import browser_policy

    restricted, mode = _policy_inputs(ctx)
    return browser_policy.browser_url_block_reason(
        url, ctx, restricted=restricted, runtime_mode=mode)


def request_reason(ctx: Any, request: dict[str, Any]) -> str:
    """Refuse missing bytes/identity; never judge a partial request as a GET."""
    if not isinstance(request, dict) or not all(_identity(request.get(k)) for k in
                                                   ("request_id", "url", "method")):
        return "BROWSER_POLICY_UNAVAILABLE: request identity, URL or method missing"
    if "post_data" not in request or not (request["post_data"] is None or isinstance(request["post_data"], str)):
        return "BROWSER_POLICY_UNAVAILABLE: complete request body fact missing"
    from ouroboros import browser_policy

    restricted, mode = _policy_inputs(ctx)
    return browser_policy.browser_request_block_reason(
        types.SimpleNamespace(url=request["url"], method=request["method"],
                              post_data=request["post_data"]), ctx,
        restricted=restricted, runtime_mode=mode)


def effect_reason(ctx: Any) -> str:
    """Unqualified request channels do not grant ordinary callers input authority."""
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
    from ouroboros.tool_access import active_tool_profile

    if active_tool_profile(ctx) == "local_readonly_subagent":
        return "BROWSER_LOCAL_READONLY_BLOCKED: this profile cannot perform browser input"
    restricted, mode = _policy_inputs(ctx)
    if restricted or not mode_has_unrestricted_agency(mode):
        return "BROWSER_REQUEST_GUARD_UNAVAILABLE: input request channels require unrestricted agency until qualified"
    return ""


def action_reason(ctx: Any, name: str, args: dict[str, Any], state: dict[str, Any]) -> str:
    """Outer invocation shape; every real primitive is checked again at the guard."""
    if name not in TOOLS:
        return "BROWSER_POLICY_UNAVAILABLE: unknown Chrome tool"
    if name == STATE_TOOL:
        return "" if not args else "BROWSER_POLICY_UNAVAILABLE: browser_state takes no arguments"
    if state["closure"]["revoked"]:
        return "BROWSER_SESSION_REVOKED: Chrome connection was revoked"
    if name == "browser_connect":
        if state["connected"] or not isinstance(args, dict) or not (
                (set(args) == {"tab_id"} and _identity(args["tab_id"]))
                or (set(args) == {"url"} and _identity(args["url"]))):
            return "BROWSER_POLICY_UNAVAILABLE: connect requires a disconnected state and exactly url or tab_id"
        if "url" in args:
            return effect_reason(ctx) or target_reason(ctx, args["url"])
        return ""
    if name == "browser_resume":
        return "" if not state["connected"] else "BROWSER_POLICY_UNAVAILABLE: resume requires a disconnected state"
    if name in {"browser_handoff", "browser_disconnect"}:
        return "" if state["connected"] else "BROWSER_SESSION_DETACHED: no connected lease"
    if not state["connected"]:
        return "BROWSER_SESSION_DETACHED: no connected lease"
    if not state["lease"]["guard_armed"]:
        return "BROWSER_REQUEST_GUARD_UNAVAILABLE: current lease is not armed"
    if name == "browser_execute":
        from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

        if not isinstance(args.get("code"), str) or not args["code"]:
            return "BROWSER_POLICY_UNAVAILABLE: execute code missing"
        restricted, mode = _policy_inputs(ctx)
        if restricted or not mode_has_unrestricted_agency(mode):
            return "BROWSER_REQUEST_GUARD_BYPASS: executor JavaScript requires unrestricted agency"
    elif name == "browser_action":
        if args.get("operation") == "sequence":
            steps = sequence_steps(args.get("args"))
            if steps is None:
                return "BROWSER_POLICY_UNAVAILABLE: sequence requires 1–32 exact operation/args steps"
            for step in steps:
                if reason := action_reason(ctx, name, step, state):
                    return reason
            return ""
        if not isinstance(args.get("operation"), str) or args["operation"] not in OPERATIONS or not isinstance(args.get("args"), dict):
            return "BROWSER_POLICY_UNAVAILABLE: action operation or args invalid"
        if args["operation"] not in READ_OPERATIONS:
            if reason := effect_reason(ctx):
                return reason
        if args["operation"] in NAVIGATIONS:
            url = args["args"].get("url")
            if not _identity(url):
                return "BROWSER_POLICY_UNAVAILABLE: navigation destination missing"
            return target_reason(ctx, url)
    return target_reason(ctx, state["lease"]["url"])


def sequence_steps(arguments: Any) -> list[dict[str, Any]] | None:
    """Finite selected mechanics, not arbitrary code or nested control calls."""
    if not isinstance(arguments, dict) or set(arguments) != {"steps"}:
        return None
    steps = arguments["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
        return None
    if any(not isinstance(step, dict) or set(step) != {"operation", "args"}
           or not isinstance(step["operation"], str) or step["operation"] not in OPERATIONS
           or not isinstance(step["args"], dict) for step in steps):
        return None
    return steps


def guard_reason(ctx: Any, facts: Any, state: dict[str, Any] | None,
                 invocation: tuple[str, str, dict[str, Any]] | None, revoked: bool,
                 primitive_index: int = 0) -> str:
    """An independent async callback: no outer call lock, no cached guard grant."""
    if revoked:
        return "BROWSER_SESSION_REVOKED: task browser session is closing"
    if not isinstance(facts, dict) or state is None:
        return "BROWSER_POLICY_UNAVAILABLE: no observed Chrome state"
    if reason := invocation_reason(facts, invocation, 0):
        return reason
    try:
        callback_state = validate_state(facts.get("state"),
                                        (state["owner"]["task_id"], str(state["owner"]["attempt"])))
    except ChromeStateError as exc:
        return str(exc)
    if lease_key(callback_state) != lease_key(state):
        return "BROWSER_LEASE_CHANGED: guard lease does not match current invocation"
    lease = state["lease"]
    _, name, outer_args = invocation
    if name in {"browser_connect", "browser_resume"}:
        if (facts.get("kind") != "primitive" or facts.get("operation") != name.removeprefix("browser_")
                or facts.get("args") != outer_args):
            return "BROWSER_POLICY_UNAVAILABLE: connection callback differs from its control call"
        if not state["connected"] or lease is None or not lease["guard_armed"]:
            return "BROWSER_REQUEST_GUARD_UNAVAILABLE: connection did not provide an armed lease"
        if name == "browser_connect" and "url" in outer_args:
            if reason := effect_reason(ctx) or target_reason(ctx, outer_args["url"]):
                return reason
        return target_reason(ctx, lease["url"])
    if not state["connected"] or lease is None or not lease["guard_armed"]:
        return "BROWSER_REQUEST_GUARD_UNAVAILABLE: no armed current lease"
    if name == "browser_execute":
        if reason := action_reason(ctx, name, outer_args, state):
            return reason
    if facts.get("kind") == "request":
        if facts.get("complete") is not True:
            return "BROWSER_POLICY_UNAVAILABLE: incomplete request facts"
        return request_reason(ctx, {key: facts.get(key) for key in
                                    ("request_id", "url", "method", "post_data")
                                    if key in facts})
    if facts.get("kind") != "primitive":
        return "BROWSER_POLICY_UNAVAILABLE: guard kind missing"
    operation, arguments = facts.get("operation"), facts.get("args")
    if not isinstance(operation, str) or operation not in OPERATIONS or not isinstance(arguments, dict):
        return "BROWSER_POLICY_UNAVAILABLE: unknown primitive or arguments"
    if name == "browser_action" and outer_args.get("operation") == "sequence":
        steps = sequence_steps(outer_args.get("args"))
        if steps is None or primitive_index >= len(steps):
            return "BROWSER_POLICY_UNAVAILABLE: sequence step is outside its selected mechanics"
        outer_args = steps[primitive_index]
    if name != "browser_execute" and (name != "browser_action" or operation != outer_args.get("operation")
                                      or arguments != outer_args.get("args")):
        return "BROWSER_POLICY_UNAVAILABLE: primitive differs from named action"
    return action_reason(ctx, "browser_action", {"operation": operation, "args": arguments}, state)


def navigation_transition(ctx: Any, prior: dict[str, Any], current: dict[str, Any]) -> str:
    """The one in-flight lease change a successful navigation step can authorize."""
    before, after = prior["lease"], current["lease"]
    if (before is None or after is None or prior["owner"] != current["owner"]
            or prior["executor_generation"] != current["executor_generation"]
            or before["tab_id"] != after["tab_id"] or not after["guard_armed"]
            or after["generation"] <= before["generation"]
            or after["lease_id"] == before["lease_id"]
            or after["document_id"] == before["document_id"]):
        return "BROWSER_LEASE_CHANGED: navigation did not advance the document lease"
    return target_reason(ctx, after["url"])


def resolve_binding(cfg: Any, ctx: Any) -> tuple[Any, str]:
    """Resolve current ready bytes; the Settings selector is not a second hash store.

    The session captures the returned hash. Subsequent checks revoke that session
    on drift; a new task resolves a newly reviewed installation without an edit
    to MCP Settings. Unbound owner-configured MCP commands retain their own path.
    """
    binding = getattr(cfg, "browser_skill_binding", None)
    if binding is None:
        return cfg, ""
    import dataclasses

    from ouroboros.config import get_skills_repo_path
    from ouroboros.marketplace.isolated_deps import augment_env_for_skill_deps
    from ouroboros.skill_loader import discover_skill_identity
    from ouroboros.skill_peer_inventory import discover_skill_peers
    from ouroboros.skill_readiness import skill_readiness_for_execution
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.tools.skill_exec import _resolve_runtime_binary, _resolve_script_path

    name, script_rel = binding
    root = canonical_data_root(ctx)
    repo = get_skills_repo_path()
    found = discover_skill_identity(root, name, repo_path=repo)
    if len(found) != 1 or found[0].name != name:
        raise ChromeStateError("BROWSER_SKILL_BINDING_REVOKED: skill identity is missing or ambiguous")
    skill = found[0]
    if skill.load_error or skill.identity_collision or not skill.manifest.is_script():
        raise ChromeStateError("BROWSER_SKILL_BINDING_REVOKED: declared script skill is unavailable")
    declared = []
    for entry in skill.manifest.scripts:
        script = entry.get("name", "")
        if isinstance(script, str):
            declared.append(script if "/" in script or script.startswith(".") else f"scripts/{script}")
    if script_rel not in declared:
        raise ChromeStateError("BROWSER_SKILL_BINDING_REVOKED: executor is not declared in the manifest")
    script = _resolve_script_path(skill.skill_dir, script_rel)
    if script is None:
        raise ChromeStateError("BROWSER_SKILL_BINDING_REVOKED: declared executor path is unavailable")
    peers = list(discover_skill_peers(root, repo_path=repo))
    readiness = skill_readiness_for_execution(root, skill, skills=peers)
    if not readiness.ready:
        raise ChromeStateError("BROWSER_SKILL_BINDING_REVOKED: skill execution readiness changed ("
                               + ", ".join(readiness.blockers) + ")")
    # Discovery just hashed this identity's complete payload; readiness judged
    # that same hash. Reuse it within this resolution, never across admissions.
    current_hash = skill.content_hash
    binary, _ = _resolve_runtime_binary(skill.manifest.runtime)
    if not binary:
        raise ChromeStateError("BROWSER_SKILL_BINDING_REVOKED: declared skill runtime is unavailable")
    env = augment_env_for_skill_deps(dict(cfg.env), skill.skill_dir)
    resolved = dataclasses.replace(cfg, command=str(binary), args=[str(script)],
                                   cwd=str(skill.skill_dir), env=env)
    return resolved, current_hash


def binding_reason(cfg: Any, ctx: Any, expected_hash: str = "") -> str:
    """Recheck canonical readiness and, for an open session, its captured bytes."""
    try:
        _resolved, current_hash = resolve_binding(cfg, ctx)
        if expected_hash and current_hash != expected_hash:
            return "BROWSER_SKILL_BINDING_REVOKED: skill payload changed"
        return ""
    except ChromeStateError as exc:
        return str(exc)
    except Exception as exc:
        logging.getLogger(__name__).warning("Chrome skill binding check failed", exc_info=True)
        return f"BROWSER_SKILL_BINDING_REVOKED: skill binding could not be verified ({type(exc).__name__})"


def lease_authority_reason(self: Any, ctx: Any) -> str:
    """Use current task owners; a retained lease is not an outstanding tool call."""
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.mcp_task_sessions import _owner
    from ouroboros.owner_pause import launch_admission
    from ouroboros.tools.registry_guards import _resource_allowed

    try:
        if not self.is_current() or _owner(ctx) != (self.key[0], self.attempt):
            return "BROWSER_SESSION_REVOKED: lease owner is no longer current"
        if not _resource_allowed(ctx, "network"):
            return "RESOURCE_CONSTRAINT_BLOCKED: lease admission requires allowed_resources.network"
        metadata = getattr(ctx, "task_metadata", None) or {}
        root_id = metadata.get("root_task_id") or getattr(ctx, "root_task_id", "") or self.key[0]
        if any(cancel_pending(self.data_root, task_id, strict=True, propagate_busy=True)
               for task_id in {self.key[0], root_id}):
            return "BROWSER_SESSION_REVOKED: task cancellation is pending"
        # No submission, retry or review-episode arming. The same live Pause
        # and sleep owners must admit this new request before it is released.
        with launch_admission(ctx, wait=False):
            return ""
    except Exception as exc:
        return f"BROWSER_POLICY_UNAVAILABLE: lease task authority ({type(exc).__name__}: {exc})"


def background_read_reason(self: Any, facts: Any, state: Any, ctx: Any) -> str:
    """One intercepted read on the existing lease; never action/navigation authority."""
    if (not isinstance(facts, dict) or facts.get("kind") != "request"
            or "invocation_id" not in facts or facts["invocation_id"] is not None):
        return "BROWSER_POLICY_UNAVAILABLE: callback invocation does not identify an idle request"
    if (facts.get("method") not in ("GET", "HEAD") or facts.get("complete") is not True
            or facts.get("has_post_data") is not False
            or "post_data" not in facts or facts["post_data"] not in (None, "")
            or not isinstance(facts.get("resource_type"), str)
            or facts["resource_type"] not in READ_RESOURCES or type(facts.get("is_main_frame")) is not bool):
        return "BROWSER_POLICY_UNAVAILABLE: background read requires a complete empty GET/HEAD subresource"
    try:
        owner = (self.key[0], self.attempt)
        validate_state(state, owner)
        candidate = validate_state(facts.get("state"), owner)
    except ChromeStateError as exc:
        return str(exc)
    lease = state["lease"]
    if (not state["connected"] or lease is None or not lease["guard_armed"]
            or lease_key(candidate) != lease_key(state)
            or facts.get("target_tab_id") != lease["tab_id"]):
        return "BROWSER_LEASE_CHANGED: background read requires the same armed tab/document lease"
    if reason := lease_authority_reason(self, ctx):
        return reason
    authority = _policy_inputs(ctx)
    reason = (target_reason(ctx, candidate["lease"]["url"])
              or request_reason(ctx, facts))
    if reason:
        return reason
    if _policy_inputs(ctx) != authority:
        return "BROWSER_POLICY_UNAVAILABLE: background actor authority changed during assessment"
    return ""


async def callback_authority_reason(self: Any, ctx: Any, authority: tuple[bool, str]) -> str:
    """The final admission shared by active and idle callbacks after policy awaits."""
    reason = await asyncio.to_thread(binding_reason, self.cfg, ctx, getattr(self, "_chrome_skill_hash", ""))
    if reason:
        self._mark_binding_revoked()
        return reason
    # Binding resolution yields; controls and actor caps must still hold after it.
    reason = await asyncio.to_thread(lease_authority_reason, self, ctx)
    if reason:
        return reason
    if self.revoked or self.ctx is not ctx or _policy_inputs(ctx) != authority:
        return "BROWSER_POLICY_UNAVAILABLE: callback actor authority changed during assessment"
    return ""


async def guard_callback(self, facts: dict[str, Any]) -> str:
    """Settle against a captured invocation or the same idle read lease, never a later call."""
    invocation = self._chrome_invocation
    previous, ctx = self._chrome_state, self.ctx
    if self.revoked:
        return "BROWSER_SESSION_REVOKED: task browser session is closing"
    if reason := await asyncio.to_thread(binding_reason, self.cfg, self.ctx, getattr(self, "_chrome_skill_hash", "")):
        self._mark_binding_revoked()
        return reason
    # A foreign, delayed or repeated callback cannot rebind a lease, even
    # if its state itself is well formed. Check this before reading state.
    if self._chrome_invocation is not invocation:
        return "BROWSER_POLICY_UNAVAILABLE: callback invocation closed during readiness check"
    authority = _policy_inputs(ctx)
    if invocation is None:
        reason = await asyncio.to_thread(background_read_reason, self, facts, previous, ctx)
        if not reason:
            reason = await callback_authority_reason(self, ctx, authority)
        if self.revoked:
            return reason or "BROWSER_SESSION_REVOKED: task browser session is closing"
        if reason:
            return reason
        if (self._chrome_invocation is not None or self._chrome_state is not previous
                or self.ctx is not ctx or not self.is_current() or _policy_inputs(ctx) != authority):
            return "BROWSER_POLICY_UNAVAILABLE: background read authority closed during assessment"
        # No invocation counters, request/document receipts or observed state
        # are advanced. In particular this cannot authorize a later document.
        return reason
    if reason := invocation_reason(facts, invocation, self._chrome_last_step):
        return reason
    try:
        candidate = validate_state(facts.get("state"), (self.key[0], self.attempt))
    except ChromeStateError as exc:
        return str(exc)
    previous = self._chrome_state
    if previous is None:
        return "BROWSER_STATE_UNKNOWN: no same-connection state observation"
    changed = lease_key(candidate) != lease_key(previous)
    url_refresh = same_document_url_refresh(previous, candidate)
    if url_refresh:
        if reason := await asyncio.to_thread(target_reason, self.ctx, candidate["lease"]["url"]):
            return reason
    if changed:
        if invocation[1] in {"browser_connect", "browser_resume"} and not previous["connected"]:
            if (not candidate["connected"] or candidate["lease"] is None
                    or candidate["executor_generation"] != previous["executor_generation"]
                    or not candidate["lease"]["guard_armed"]
                    or candidate["lease"]["generation"] <= self._chrome_lease_generation_floor
                    or (invocation[1] == "browser_connect"
                        and "tab_id" in invocation[2]
                        and candidate["lease"]["tab_id"] != invocation[2].get("tab_id"))):
                return "BROWSER_LEASE_CHANGED: connection owner, executor or tab changed"
        elif (facts.get("kind") == "document"
              and isinstance(candidate["lease"], dict)
              and facts.get("request_id") in getattr(self, "_chrome_documents", {})
              and self._chrome_documents[facts["request_id"]] == candidate["lease"]["url"]):
            if reason := await asyncio.to_thread(navigation_transition, self.ctx, previous, candidate):
                return reason
        else:
            return "BROWSER_LEASE_CHANGED: unexpected document or executor change"
    if facts.get("kind") == "document":
        reason = "" if changed else "BROWSER_LEASE_CHANGED: document callback did not change the lease"
    else:
        reason = await asyncio.to_thread(
            guard_reason, self.ctx, facts, candidate, invocation, self.revoked,
            getattr(self, "_chrome_sequence_index", 0))
    if not reason:
        reason = await callback_authority_reason(self, ctx, authority)
    if self.revoked:
        return reason or "BROWSER_SESSION_REVOKED: task browser session is closing"
    if self._chrome_invocation is not invocation or self._chrome_state is not previous:
        return "BROWSER_POLICY_UNAVAILABLE: callback authority closed during assessment"
    if stale := invocation_reason(facts, invocation, self._chrome_last_step):
        return stale
    if reason:
        return reason
    if facts.get("kind") == "request":
        request_id = facts["request_id"]
        if request_id in self._chrome_requests:
            return "BROWSER_POLICY_UNAVAILABLE: duplicate request identity"
        self._chrome_requests.add(request_id)
        if (self._chrome_navigation_pending and facts.get("resource_type") == "Document"
                and facts.get("is_main_frame") is True
                and facts.get("target_tab_id") == previous["lease"]["tab_id"]):
            self._chrome_documents[request_id] = facts["url"]
    self._chrome_last_step = facts["step_id"]
    if (facts.get("kind") == "primitive" and invocation[1] == "browser_action"
            and invocation[2].get("operation") == "sequence"):
        self._chrome_sequence_index = getattr(self, "_chrome_sequence_index", 0) + 1
    if changed or url_refresh:
        self._chrome_state = candidate
        if changed:
            self._chrome_navigation_pending = False
        if candidate["lease"] is not None:
            self._chrome_lease_generation_floor = candidate["lease"]["generation"]
    if (facts.get("kind") == "primitive"
            and facts.get("operation") in NAVIGATIONS | INPUT_OPERATIONS):
        self._chrome_navigation_pending = True
    return ""


def observe_state(self: Any, result: Any, *, allow_closed: bool = False) -> dict[str, Any]:
    """Refresh same-document URLs; require a witnessed transition for a new lease."""
    state = parse_state(result,
                        (self.key[0], self.attempt), allow_closed=allow_closed)
    prior = self._chrome_state
    if prior is not None:
        if state["executor_generation"] != prior["executor_generation"]:
            raise RuntimeError("BROWSER_LEASE_CHANGED: Chrome executor generation changed")
        if state["connected"]:
            if not prior["connected"] or lease_key(state) != lease_key(prior):
                raise RuntimeError("BROWSER_LEASE_CHANGED: Chrome lease changed without a guard callback")
            if same_document_url_refresh(prior, state):
                if not state["lease"]["guard_armed"]:
                    raise ChromeStateError("BROWSER_REQUEST_GUARD_UNAVAILABLE: current lease is not armed")
                if reason := target_reason(self.ctx, state["lease"]["url"]):
                    raise ChromeStateError(reason)
        elif prior["connected"] and not allow_closed:
            raise RuntimeError("BROWSER_SESSION_DETACHED: Chrome disconnected outside a control call")
    lease = state["lease"]
    if lease is not None:
        if lease["generation"] < self._chrome_lease_generation_floor:
            raise RuntimeError("BROWSER_LEASE_CHANGED: Chrome lease generation moved backwards")
        if (lease["generation"] == self._chrome_lease_generation_floor
                and self._chrome_state is not None and self._chrome_state["lease"] is not None
                and lease_key(state) != lease_key(self._chrome_state)):
            raise RuntimeError("BROWSER_LEASE_CHANGED: Chrome lease reused a generation")
        self._chrome_lease_generation_floor = lease["generation"]
    self._chrome_state = state
    return state



def observe_before_call(self: Any, name: str, timeout: int) -> dict[str, Any]:
    """Only a positively clean explicit disconnect grants connect after an old lease."""
    from ouroboros.mcp_task_sessions import BrowserBridgeRefusal

    receipt = getattr(self, "_chrome_clean_disconnect", None)
    prior = self._chrome_state
    reconnect = (name == "browser_connect" and prior is not None and not prior["connected"]
                 and self._chrome_lease_generation_floor >= 0)
    if reconnect:
        if receipt is None or lease_key(prior) != receipt:
            raise BrowserBridgeRefusal("BROWSER_SESSION_DETACHED: connect requires a clean explicit disconnect")
        if reason := lease_authority_reason(self, self.ctx):
            raise BrowserBridgeRefusal(reason)
    try:
        state = self._observe_before(timeout, allow_closed=name in {STATE_TOOL, "browser_resume"} or reconnect)
    except Exception:
        self._chrome_clean_disconnect = None
        raise
    if receipt is not None and lease_key(state) != receipt:
        self._chrome_clean_disconnect = None
        if reconnect:
            raise BrowserBridgeRefusal("BROWSER_LEASE_CHANGED: disconnected state changed before connect")
    return state


def call_chrome(self, prefixed: str, name: str, args: dict, ctx: Any,
                 timeout: int) -> tuple[ToolResult, str, str]:
    from ouroboros.mcp_client import _is_task_control, _tool_result_from_call_result
    from ouroboros.mcp_task_sessions import BrowserBridgeRefusal, _chrome_binding_reason, _owner
    from ouroboros.safety import check_safety

    with self._call_lock:
        if self.revoked or _owner(ctx) != (self.key[0], self.attempt):
            raise BrowserBridgeRefusal("BROWSER_SESSION_REVOKED: Chrome owner changed or was revoked")
        if reason := _chrome_binding_reason(self.cfg, ctx, getattr(self, "_chrome_skill_hash", "")):
            self.stop()
            raise BrowserBridgeRefusal(reason)
        self.ctx = ctx
        state = observe_before_call(self, name, timeout)
        reason = action_reason(ctx, name, args, state)
        if reason:
            raise BrowserBridgeRefusal(reason)
        lease = state["lease"]
        binding = {"browser_page": {
            "url": lease["url"] if lease else "",
            "tab_id": lease["tab_id"] if lease else args.get("tab_id", ""),
            "lease_id": lease["lease_id"] if lease else "",
            "observed_by": "browser_state, same connection",
        }}
        allowed, advice = check_safety(
            prefixed, args, messages=getattr(ctx, "messages", None), ctx=ctx,
            resolved_binding=binding)
        if not allowed:
            raise BrowserBridgeRefusal(advice, code="SAFETY_VIOLATION")
        if observe_before_call(self, name, timeout) != state:
            raise BrowserBridgeRefusal(
                "BROWSER_STATE_CHANGED: Chrome lease changed during assessment; action not dispatched")
        if reason := _chrome_binding_reason(self.cfg, ctx, getattr(self, "_chrome_skill_hash", "")):
            self.stop()
            raise BrowserBridgeRefusal(reason)
        if reason := action_reason(ctx, name, args, state):
            raise BrowserBridgeRefusal(reason)
        invocation_id = uuid.uuid4().hex
        self._chrome_invocation = (invocation_id, name, args)
        self._chrome_last_step = 0
        self._chrome_sequence_index = 0
        self._chrome_requests.clear()
        self._chrome_documents.clear()
        self._chrome_navigation_pending = False
        disconnect_receipt = getattr(self, "_chrome_clean_disconnect", None)
        self._chrome_clean_disconnect = None
        try:
            raw = self._request(name, args, timeout, dispatch=True, invocation_id=invocation_id)
            from ouroboros.mcp_task_sessions import _retain_returned_response

            source, retention_gap = _retain_returned_response(self, raw)
        finally:
            self._chrome_invocation = None
            self._chrome_navigation_pending = False
        blocked = [self._blocked.popleft() for _ in range(len(self._blocked))]
        try:
            result = _tool_result_from_call_result(raw)
            after = self._observe(timeout, allow_closed=name in {STATE_TOOL, "browser_disconnect", "browser_handoff"})
            reason = action_reason(ctx, "browser_state", {}, after)
            if not reason and after["connected"]:
                from ouroboros.mcp_chrome_protocol import target_reason

                reason = target_reason(ctx, after["lease"]["url"])
                if not after["lease"]["guard_armed"]:
                    reason = reason or "BROWSER_REQUEST_GUARD_UNAVAILABLE: new lease is not armed"
        except Exception as exc:
            if _is_task_control(exc):
                raise
            # The action already returned. SDK/projection/state failures must
            # keep that response source and reach the manager's returned accounting.
            reason = f"{type(exc).__name__}: {exc}"
        if reason:
            from ouroboros.mcp_task_sessions import _post_dispatch_error

            result = _post_dispatch_error(name, reason, source, retention_gap)
        elif result.status == "ok" and not self.revoked:
            if (name == "browser_disconnect" and not after["connected"]
                    and after["closure"] == {"revoked": False, "detached": True}):
                self._chrome_clean_disconnect = lease_key(after)
            elif name == STATE_TOOL and disconnect_receipt == lease_key(after):
                self._chrome_clean_disconnect = disconnect_receipt
        note = ("⚠️ BROWSER_REQUEST_BLOCKED: Chrome guard refusals since the previous response:\n"
                + "\n".join(f"- {line}" for line in blocked)) if blocked else ""
        if self._blocked_omitted:
            note += f"\nEarlier guard diagnostics omitted: {self._blocked_omitted}."
            self._blocked_omitted = 0
        return result, advice, note


def normalize_browser_entry(raw: dict[str, Any], transport: str) -> tuple[bool, str, tuple[str, str] | None]:
    """Normalize the explicit browser protocol and canonical skill selector."""
    browser_bridge = raw.get("browser_bridge", False)
    if not isinstance(browser_bridge, bool) or (browser_bridge and transport != "stdio"):
        raise ValueError("browser_bridge must be a boolean and requires stdio")
    browser_protocol = raw.get("browser_protocol", "playwright")
    if browser_protocol not in ("playwright", "ouroboros_chrome"):
        raise ValueError("browser_protocol must be playwright or ouroboros_chrome")
    if browser_protocol == "ouroboros_chrome" and (transport != "stdio" or not browser_bridge):
        raise ValueError("ouroboros_chrome requires stdio and browser_bridge")
    binding_raw = raw.get("browser_skill_binding")
    browser_skill_binding = None
    if binding_raw is not None:
        if browser_protocol != "ouroboros_chrome" or not isinstance(binding_raw, dict) or set(binding_raw) != {
            "skill_name", "script_path"
        }:
            raise ValueError("browser_skill_binding requires ouroboros_chrome and exact skill_name, script_path")
        skill_name, script_path = (
            binding_raw[key] for key in ("skill_name", "script_path"))
        if (not isinstance(skill_name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", skill_name)
                or not isinstance(script_path, str) or not script_path or "\\" in script_path
                or script_path.startswith(("/", "~")) or any(part in ("", ".", "..") for part in script_path.split("/"))):
            raise ValueError("browser_skill_binding has an invalid identity or declared script path")
        browser_skill_binding = (skill_name, script_path)
    return browser_bridge, browser_protocol, browser_skill_binding
