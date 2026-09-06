from __future__ import annotations

from ouroboros.extension_loader import PluginAPIImpl, _PluginAPIConfig


def test_skill_job_dir_creates_isolated_job_tree(tmp_path):
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=[],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
    ))

    job_dir = api.skill_job_dir("scene/001:retry")

    assert job_dir.name.startswith("scene_001_retry-")
    assert (job_dir / "assets").is_dir()
    assert (job_dir / "output").is_dir()
    assert (job_dir / "tmp").is_dir()
    assert api.skill_job_dir("scene/001:retry") == job_dir
    assert api.skill_job_dir("scene_001_retry") != job_dir


def test_skill_job_dir_sanitizes_empty_and_long_ids(tmp_path):
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=[],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
    ))

    assert api.skill_job_dir("...").name.startswith("_job-")
    assert len(api.skill_job_dir("x" * 100).name) == 64
    assert api.skill_job_dir(("x" * 64) + "a") != api.skill_job_dir(("x" * 64) + "b")
