"""Filename shapes that commonly indicate credential material (leaf module).

Single source for the credential-name dictionaries and the name regex that
previously lived as three inline copies (tool_access + two in tools/core).

Consumers: MUTATION-side user_files gates (write/edit/shell targets), the
Deliverables/output lexical guards, attachment-staging ingest, and the child
(subagent) location-deny paths. The ROOT principal's READ authorization must
NOT consult these shapes (capinv-447 / В23: the owner's root agent reads the
owner's home in full; secret BYTES are masked at egress instead) — an
import-boundary test pins that ``ouroboros.tool_access`` resolves root read
decisions without importing this module.
"""

from __future__ import annotations

import pathlib
import re

# Directory components whose contents are treated as credential/control stores.
CREDENTIAL_COMPONENT_NAMES = frozenset({
    ".aws",
    ".azure",
    ".config",
    ".docker",
    ".git",   # v6.52.0: VCS internals hold config + stored credentials
    ".gnupg",
    ".hg",
    ".kube",
    ".local",
    ".netrc",
    ".ssh",
    ".svn",
    "library",
})

# Credential / shell-init / history file names (lowercase).
CREDENTIAL_FILE_NAMES = frozenset({
    ".env",
    ".bash_history",
    ".bash_profile",
    ".bashrc",
    ".dockercfg",
    ".git-credentials",
    ".gitconfig",
    ".htpasswd",
    ".npmrc",
    ".pgpass",
    ".profile",
    ".pypirc",
    ".python_history",
    ".zsh_history",
    ".zprofile",
    ".zshrc",
    "auth.json",
    "credentials",
    "credentials.json",
    "secrets.json",
    "settings.json",
    "token.json",
    "tokens.json",
})

# v6.52.0 (P1): a SMALL allowlist of benign hidden (dot) project components used
# by the DEFAULT-DENY dotted-component rule on mutation/output surfaces: a
# credential blocklist can never be exhaustive (~/.terraform.d, ~/.cargo,
# ~/.pip, ...), so a dotted component is refused UNLESS it is a known-safe
# project-config dir/file.
BENIGN_DOT_NAMES = frozenset({
    ".github",
    ".gitlab",
    ".circleci",
    ".devcontainer",
    ".vscode",
    ".idea",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".dockerignore",
    ".editorconfig",
})

# Names denied to restricted (read-only / acting) subagents on the repo/data
# drives — the child location-deny vocabulary (tools/core.py).
SUBAGENT_CREDENTIAL_FILE_NAMES = frozenset({
    ".env",
    ".netrc",
    "auth.json",
    "credentials",
    "credentials.json",
    "keys.json",
    "secret.json",
    "secrets.json",
    "settings.json",
    "settings.json.lock",
    "token.json",
    "tokens.json",
})

CREDENTIAL_NAME_RE = re.compile(
    r"(?:^|[._-])(api[_-]?key|credential|password|secret|token)(?:[._-]|$)", re.I
)

CREDENTIAL_FILE_SUFFIXES = (".key", ".pem", ".p12", ".pfx")


def user_files_mutation_shape_reason(resolved: pathlib.Path, home: pathlib.Path) -> str:
    """Shape gate for MUTATING user_files operations (write/edit/shell targets).

    Keeps the pre-capinv-447 hidden/credential semantics: DEFAULT-DENY dotted
    components (minus the benign project allowlist) plus credential-like names.
    Never applied to root read/list/search — those are location-authorized only.
    Empty string = no objection.
    """
    try:
        parts = resolved.relative_to(home).parts
    except ValueError:
        parts = resolved.parts
    for part in parts:
        if not part:
            continue
        part_lower = part.lower()
        if part_lower in CREDENTIAL_COMPONENT_NAMES:
            return "path is hidden or credential-like (secret/credential directory)"
        if part.startswith(".") and part_lower not in BENIGN_DOT_NAMES:
            return "path is hidden or credential-like (non-allowlisted hidden component)"
    name = resolved.name
    name_lower = name.lower()
    if (
        name_lower in CREDENTIAL_FILE_NAMES
        or CREDENTIAL_NAME_RE.search(name)
        or name_lower.endswith(CREDENTIAL_FILE_SUFFIXES)
    ):
        return "path name is credential-like"
    return ""
