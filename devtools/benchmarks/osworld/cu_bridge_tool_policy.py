"""Host/skill tool policy for the OSWorld cu_bridge runner.

Verbatim extraction from ``run_cu_bridge_agent.py`` (v7 stream W): the allowlist
of core tools the untrusted OSWorld task legitimately needs, the computed host
denylist, the GUI action tools counted for the budget disclosure, and the
connection-management skill surfaces the benchmark agent must not reach.
"""

from __future__ import annotations

from pathlib import Path

from devtools.benchmarks.osworld.cu_bridge_runtime import SKILL_NAME

# The OSWorld task instruction is UNTRUSTED and the VM is driven ONLY through the
# unix_computer_use skill (ext_* tools). Rather than a fragile per-tool DENYLIST
# (which silently misses any host tool added later), the runner keeps a small
# ALLOWLIST of core tools the task legitimately needs and DENIES every other core
# tool — so any host execution/mutation/VCS/GitHub/service/self-mod/chat surface,
# present or future, is blocked by construction. The skill's ext_* tools are not
# core tools, so they are never on the computed denylist and always available.
# `enable_tools` is kept (the agent must enable the computer-use skill), which in
# principle could enable OTHER extensions — but the runner seeds and enables ONLY
# unix_computer_use into a FRESH isolated bench data dir (append-only per task per
# the runbook), so there is no other extension to reach; a reused multi-extension
# data dir is out of the supported bench setup.
# Deliberately NO host filesystem/code read tools (read_file/list_files/search_code/
# query_code): the isolated bench settings.json holds provider API keys, and a
# prompt-injected task is a normal root task that could read_file(root="runtime_data",
# "settings.json") to exfiltrate them. The agent inspects the VM through the skill
# (remote_exec/screenshot), never the host filesystem.
_ALLOWED_CORE_TOOLS = frozenset({
    "list_available_tools", "enable_tools",   # discover + enable the computer-use skill
    "view_image",                             # the vision channel (SEE screenshots)
    "compact_context", "set_tool_timeout",    # agent self-management (no host access)
})


def _core_tool_names() -> set[str]:
    """All built-in (non-extension) core tool names, for the computed denylist."""
    import tempfile

    from ouroboros.tools.registry import ToolRegistry

    tmp = Path(tempfile.mkdtemp(prefix="cu_bridge_toolscan_"))
    reg = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    return {t["function"]["name"] for t in reg.schemas()}


def _host_denied_tools() -> list[str]:
    """Deny every core tool the OSWorld task does not need (allowlist-complement)."""
    return sorted(_core_tool_names() - _ALLOWED_CORE_TOOLS)


# GUI action tools (short skill names) counted for the budget disclosure.
_GUI_ACTION_TOOLS = frozenset({
    "click", "move", "left_click_drag", "mouse_down", "mouse_up",
    "type_text", "key", "hold_key", "scroll",
    # v6.81.1: the skill registers these as thin click aliases. They are the same
    # mutating surface under other names — leaving them out of this set let the
    # "cannot act by construction" premise phase click the VM through an alias
    # and under-counted gui_action_calls in the disclosure counters (caught by
    # both triad reviewers on the release diff). Any future click alias MUST be
    # added here in the same commit that registers it.
    "double_click", "triple_click",
})


# unix_computer_use ext tools the untrusted task must NOT reach. The runner pins
# the active connection to the published OSWorld VM; a task that could switch the
# backend (use_local/activate_connection local) or retarget it (add_connection)
# would drive the HOST desktop instead — defeating the host lockdown AND the
# fail-closed guarantee. Read-only introspection (list_connections/test_connection)
# stays; the mutating connection-management surface is denied.
# Connection-management surfaces the benchmark agent must not reach. `list_connections`
# and `test_connection` join the mutating ones (v6.81.1): both echo the bridge URL, which
# is control-plane. A v6.81.1 trace shows why that matters — an agent that learned the
# port from a tool result went looking for `<bridge>/evaluate`, i.e. for the grader. The
# runner pins the connection itself, so the agent never needs either tool.
_DENIED_SKILL_EXT_TOOLS = ("add_connection", "activate_connection", "use_local",
                           "clear_active_connection", "list_connections", "test_connection")


def _effective_disabled_tools(allow_a11y: bool, *, gate_phase: bool = False) -> list[str]:
    """Per-task disabled-tool list = the host-tool complement of the allowlist,
    plus the skill's connection-switching ext tools (the runner pins the VM
    connection), plus ``ax_tree`` unless ``--allow-a11y`` is given (screenshot-only
    by default; enabling it must disclose "a11y tree used"). ext names must be the
    provider-safe full surface names — disabled_tools matches exact names."""
    from ouroboros.extension_loader import extension_surface_name

    disabled = _host_denied_tools()
    disabled += [extension_surface_name(SKILL_NAME, t) for t in _DENIED_SKILL_EXT_TOOLS]
    if not allow_a11y:
        disabled.append(extension_surface_name(SKILL_NAME, "ax_tree"))
    disabled.append("schedule_subagent")  # operator 2026-07-23: subagents=0 no-swarm campaign
    if gate_phase:
        # Closes the GUI vector only, and says so. The mutating GUI surface is ABSENT rather
        # than discouraged, so the premise cannot be manufactured through it. remote_exec
        # stays available for read-only probes and is read-only BY INSTRUCTION ONLY —
        # classifying a shell command as reading or writing in code would be the pattern
        # gate the constitution forbids for a semantic decision. So the shell remains as
        # advisory here as it is everywhere else: this phase makes manufacturing harder,
        # not impossible, and the working phase is re-reset afterwards precisely because
        # this guarantee is partial.
        disabled += [extension_surface_name(SKILL_NAME, t) for t in sorted(_GUI_ACTION_TOOLS)]
    return disabled


_COMPUTER_USE_SHORT_TOOLS = (
    "list_connections", "test_connection", "screenshot", "click", "move",
    "left_click_drag", "mouse_down", "mouse_up", "type_text", "key", "hold_key",
    "scroll", "wait", "window_list", "ax_tree", "cursor_position", "remote_exec",
)
