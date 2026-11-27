"""A skill's ``sign_in`` declaration (``ouroboros/service_sign_in_declaration.py``) and its preflight finding."""

from __future__ import annotations

import pytest

from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
from ouroboros.service_sign_in_declaration import parse
from ouroboros.tools.skill_preflight import _widget_schema_findings

GOOD = {"id": "google", "label": "Google", "issuer": "https://accounts.example", "scopes": ["openid", "email"],
        "optional_scopes": [{"scope": "extra", "label": "Extra", "default": True}],
        "authorize_params": {"access_type": "offline"}, "accounts": "many", "optional": True,
        "notes": {"refresh_rejected": "Testing apps end sign-ins after 7 days."}}


def test_a_valid_declaration_reads_whole():
    services, problems = parse([GOOD, {"id": "whoop", "label": "WHOOP", "scopes": ["read"], "endpoints": {
        "authorization": "https://api.example/oauth/auth", "token": "https://api.example/oauth/token",
        "identity": "https://api.example/oauth/identity"}}])
    assert not problems and set(services) == {"google", "whoop"}
    assert services["google"].declared_scopes() == {"openid", "email", "extra"}
    assert services["whoop"].endpoints["identity"] == "https://api.example/oauth/identity"
    assert services["whoop"].identity == {"subject": "sub", "label": "email"}
    named, problems = parse([{**GOOD, "identity": {"subject": "id", "label": "login"}}, {**GOOD, "id": "plain", "identity": {}}])
    assert not problems and named["google"].identity == {"subject": "id", "label": "login"}
    assert named["plain"].identity == {"subject": "sub", "label": "email"}  # an explicit empty mapping is the default one


@pytest.mark.parametrize("change", [
    {"client_id": "x"}, {"client_secret": "x"}, {"issuer": "http://accounts.example"},
    {"authorize_params": {"scope": "all"}}, {"authorize_params": {"redirect_uri": "https://elsewhere.example"}},
    {"id": "Google!"}, {"scopes": []}, {"accounts": "several"}, {"notes": {"other": "x"}}, {"issuer": ""},
    {"identity": {"verified": "email_verified"}}, {"identity": []}, {"identity": ""}, {"identity": False},
])
def test_an_invalid_entry_is_left_out_with_its_problem(change):
    services, problems = parse([{**GOOD, **change}])
    assert not services and problems and problems[0].startswith("sign_in ")


def test_preflight_names_the_problems_of_a_declaration_and_passes_a_valid_one(tmp_path):
    def manifest(sign_in: str):
        return parse_skill_manifest_text(f"---\nname: cal\ndescription: d\nversion: 0.1\ntype: extension\n"
                                         f"entry: plugin.py\npermissions: []\n{sign_in}\n---\n# cal\n")

    import json
    good = [row for row in _widget_schema_findings(tmp_path, manifest("sign_in: " + json.dumps([GOOD])))
            if row["item"] == "sign_in_schema"]
    bad = [row for row in _widget_schema_findings(tmp_path, manifest("sign_in: " + json.dumps([{**GOOD, "client_secret": "s"}])))
           if row["item"] == "sign_in_schema"]
    assert good[0]["ok"] is True and bad[0]["ok"] is False and "credentials" in bad[0]["detail"]
    assert not [row for row in _widget_schema_findings(tmp_path, manifest("")) if row["item"] == "sign_in_schema"]
