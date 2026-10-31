"""A connector supplies geometry, never executable SVG or a mutable asset URL."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from ouroboros import extension_registry_state as registry
from ouroboros.extension_contributions import stage_contributions
from ouroboros.gateway.contributions import api_extension_host_asset
from ouroboros.icon_geometry import MAX_ICON_BYTES, sanitize_svg, icon_spec
from ouroboros.skill_loader import load_skill

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 2L22 22"/></svg>'
PLACE = {"id": "place", "provides": "place", "label": "Service", "destination": "vendor:Service"}


def test_geometry_is_rebuilt_in_house_style():
    clean = sanitize_svg(SVG.replace('<path d="M2 2L22 22"/>',
        '<g transform="translate(1 2)"><circle cx="12" cy="12" r="2"/><ellipse cx="3" cy="4" rx="2" ry="1"/>'
        '<rect x="1" y="1" width="2" height="3"/><line x1="0" y1="1" x2="3" y2="4"/>'
        '<polygon points="1,2 3,4 5,6"/><polyline points="1,2 3,4"/></g>'))
    assert 'viewBox="0 0 24 24"' in clean and 'stroke-width="2"' in clean
    assert 'stroke-linecap="round"' in clean and 'stroke="currentColor"' in clean
    assert "<g" in clean and "<polygon" in clean
    assert sanitize_svg(clean) == clean
    assert icon_spec("mail") == {"kind": "host", "name": "mail"}
    assert icon_spec("ui/icon.svg", SVG)["svg"] == sanitize_svg(SVG)


@pytest.mark.parametrize("body", [
    '<script>alert(1)</script>', '<style>path{fill:red}</style>', '<foreignObject/>', '<use href="#a"/>',
    '<image href="https://example.test"/>', '<path d="M1 2" onload="alert(1)"/>',
    '<path d="M1 2" href="#a"/>', '<path d="M1 2" fill="url(#a)"/>',
    '<path d="M1 2" style="stroke:red"/>', '<path d="M1 2" fill="#000"/>',
    '<path d="M1 2" stroke="red"/>', '<path d="not a path"/>', '<circle cx="NaN"/>',
    '<g transform="url(x)"/>', '<path xmlns="urn:foreign" d="M1 2"/>',
])
def test_active_non_geometry_and_colors_refused_with_valid_neighbor(body):
    with pytest.raises(ValueError):
        sanitize_svg(f'<svg viewBox="0 0 24 24">{body}</svg>')
    assert "currentColor" in sanitize_svg(SVG)


@pytest.mark.parametrize("source", [b"x" * (MAX_ICON_BYTES + 1), b"<!DOCTYPE svg><svg/>",
    b"<svg>", b"\xff", '<svg viewBox="0 0 0 24"><path d="M1 2"/></svg>',
    '<svg>' + '<path d="M1 2"/>' * 128 + '</svg>', '<svg>' + '<g>' * 17 + '</g>' * 17 + '</svg>'])
def test_physical_icon_limits(source):
    with pytest.raises(ValueError):
        sanitize_svg(source)
    assert icon_spec("ui/icon.svg", source) == {}
    assert icon_spec("ui/icon.svg", SVG)["kind"] == "svg"


def staged_icon(tmp_path, *, entries=None, icon="ui/icon.svg"):
    folder = tmp_path / "skills" / "native" / "connector"
    (folder / "ui").mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api): pass\n")
    (folder / "skill.json").write_text(json.dumps({"name": "connector", "type": "extension", "entry": "plugin.py",
        "permissions": ["presence"], "icon": icon, "contributes": entries if entries is not None else [PLACE],
        "tools": [{"name": "send", "effect": "message", "target": ["to", "cc", "bcc"]}]}))
    (folder / "ui/icon.svg").write_text(SVG)
    skill = load_skill(folder, tmp_path)
    api = SimpleNamespace(_staged=SimpleNamespace())
    stage_contributions(api, skill, skill.content_hash)
    return skill, api._staged


def test_bad_manifest_icon_falls_back_without_hiding_the_place(tmp_path):
    _, staged = staged_icon(tmp_path, icon="https://example.test/icon.svg")
    assert staged.contributions[0]["label"] == "Service"
    assert staged.contributions[0]["icon_spec"] == {}
    assert staged.contribution_problems == ["icon: icon must be a host icon name or ui/icon.svg"]


@pytest.mark.parametrize("entries", [[PLACE], []])
def test_icon_stages_without_host_page_and_pins_reviewed_bytes(tmp_path, monkeypatch, entries):
    skill, staged = staged_icon(tmp_path, entries=entries)
    assert staged.contribution_problems == []
    assert staged.host_sources["ui/icon.svg"] == sanitize_svg(SVG).encode()
    bundle = registry._ExtensionRegistrations(content_hash=skill.content_hash, host_sources=staged.host_sources)
    monkeypatch.setitem(registry._extensions, skill.name, bundle)
    (skill.skill_dir / "ui/icon.svg").write_text('<svg><script>bad()</script></svg>')
    assert registry.live_host_asset(skill.name, skill.content_hash, "ui/icon.svg") == sanitize_svg(SVG).encode()
    assert registry.live_host_asset(skill.name, "different", "ui/icon.svg") is None
    stale = SimpleNamespace(_staged=SimpleNamespace())
    stage_contributions(stale, skill, skill.content_hash)
    assert "ui/icon.svg" not in stale._staged.host_sources
    assert any("changed" in reason for reason in stale._staged.contribution_problems)
    if entries:
        row = staged.contributions[0]
        assert row["icon_spec"] == {"kind": "svg", "svg": sanitize_svg(SVG)}
        assert row["tools"][0]["name"].endswith("_send")
        assert row["tools"][0]["target"] == ["to", "cc", "bcc"]


def test_svg_asset_route_sandbox_and_other_assets_unchanged(monkeypatch):
    bundle = registry._ExtensionRegistrations(content_hash="reviewed", host_sources={"ui/icon.svg": SVG.encode(), "ui/data.json": b"{}"})
    monkeypatch.setitem(registry._extensions, "connector", bundle)
    def get(path):
        request = Request({"type": "http", "path_params": {"skill": "connector", "revision": "reviewed", "path": path}})
        return asyncio.run(api_extension_host_asset(request))
    response = get("ui/icon.svg")
    assert response.status_code == 200 and "sandbox" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in get("ui/data.json").headers
