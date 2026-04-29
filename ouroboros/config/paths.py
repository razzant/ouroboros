"""Local-first path configuration. No Google Drive references.

This module is the single source of truth for all filesystem paths.
All modules import from here instead of hardcoding paths.
"""
import os
import pathlib

# Local root: ~/.ouroboros
LOCAL_ROOT = pathlib.Path(os.environ.get("OUROBOROS_LOCAL_ROOT", str(pathlib.Path.home() / ".ouroboros")))

# Drive root: configurable, defaults to LOCAL_ROOT
DRIVE_ROOT = pathlib.Path(os.environ.get("OUROBOROS_DRIVE_ROOT", str(LOCAL_ROOT)))

# Repo root: parent of this module's directory
_REPO_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
REPO_ROOT = pathlib.Path(os.environ.get("OUROBOROS_REPO_DIR", str(_REPO_DIR)))


def drive_root() -> pathlib.Path:
    """Return the drive root path."""
    return DRIVE_ROOT


def repo_root() -> pathlib.Path:
    """Return the repository root path."""
    return REPO_ROOT


def local_root() -> pathlib.Path:
    """Return the local root path (~/.ouroboros)."""
    return LOCAL_ROOT


def drive_path(rel: str) -> pathlib.Path:
    """Get a path under the drive root."""
    return (DRIVE_ROOT / rel).resolve()


def repo_path(rel: str) -> pathlib.Path:
    """Get a path under the repo root."""
    return (REPO_ROOT / rel).resolve()
