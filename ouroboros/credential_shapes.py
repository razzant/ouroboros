"""Credential leaf shapes and physical owner locations (leaf module).

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

# Conservative directory shapes for capture/ingest, not root authorization.
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
    "auth_token.json",
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


def owner_credential_locations(home: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """The existing host credential locations, independent of project names.

    SSH's ordinary config is a file-level exception; a symlink must still be
    judged by its destination and cannot turn a key into an ordinary config.
    """
    protected = [home / rel for rel in (
        ".ssh", ".aws", ".gnupg", ".netrc", ".pgpass", ".config/gcloud",
        ".docker/config.json", ".kube/config", ".npmrc", "file1.txt",
    )]
    config = home / ".ssh" / "config"
    allowed = [config] if not config.is_symlink() and not config.is_dir() else []
    return protected, allowed


def user_files_mutation_shape_reason(resolved: pathlib.Path, home: pathlib.Path) -> str:
    """Keep credential/control writes fenced without banning ordinary configs.

    A project directory named auth, .config or Library is not host authority.
    Known credential leaves and VCS internals retain their own protection;
    root read authorization never calls this mutation-only predicate.
    """
    resolved = pathlib.Path(resolved).resolve(strict=False)
    home = pathlib.Path(home).resolve(strict=False)
    protected, allowed = owner_credential_locations(home)
    if resolved not in allowed and any(resolved.is_relative_to(path.resolve(strict=False)) for path in protected):
        return "path is hidden or credential-like (owner credential location)"
    try:
        parts = resolved.relative_to(home).parts
    except ValueError:
        parts = resolved.parts
    if any(part.lower() in {".git", ".hg", ".svn"} for part in parts):
        return "path is hidden or credential-like (VCS control directory)"
    name_lower = resolved.name.lower()
    if name_lower in (CREDENTIAL_FILE_NAMES - {"settings.json"}) or name_lower.endswith(CREDENTIAL_FILE_SUFFIXES):
        return "path name is credential-like"
    return ""
