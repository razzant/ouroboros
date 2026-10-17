"""Which meeting service a link belongs to, and what its participant probe has shown.

The vendor is a routing fact read from the link's host, never guessed from page
content. Every vendor is offered; the participant is experimental until a probe
with a real meeting passed. The shipped record (``vendor_probes.json`` beside
this skill) holds what was proven before release; an owner-run probe on this
computer writes ``vendor_probes.json`` in the skill's state directory. The join
result carries that fact, so the mind knows whether to expect listen-only.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

PROBES_FILE = "vendor_probes.json"
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"

# vendor id -> (owner-facing name, hosts whose subdomains also belong to it)
VENDORS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "zoom": ("Zoom web", ("zoom.us", "zoom.com", "zoomgov.com")),
    "meet": ("Google Meet", ("meet.google.com",)),
    "teams": ("Microsoft Teams", ("teams.microsoft.com", "teams.live.com", "teams.cloud.microsoft")),
}
SKILL_DIR = pathlib.Path(__file__).resolve().parent.parent


class VendorRefused(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def vendor_for(url: str) -> Tuple[str, str]:
    """``(vendor id, origin)`` for an https meeting link of a known service."""
    parts = urlsplit((url or "").strip())
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise VendorRefused("MEETING_LINK_INVALID", "A meeting link must be an https:// address.")
    for vendor, (_name, domains) in VENDORS.items():
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            port = f":{parts.port}" if parts.port else ""
            return vendor, f"https://{host}{port}"
    names = ", ".join(name for name, _domains in VENDORS.values())
    raise VendorRefused("MEETING_VENDOR_UNKNOWN",
                        f"{host} is not a meeting service the browser participant knows ({names}).")


def belongs_to(vendor: str, url: str) -> bool:
    host = (urlsplit(url or "").hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in VENDORS.get(vendor, ("", ()))[1])


def _read(path: pathlib.Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def probe_table(state_dir: Optional[pathlib.Path], skill_dir: pathlib.Path = SKILL_DIR) -> Dict[str, Dict[str, Any]]:
    """Per vendor: the shipped record, overridden by this computer's owner-run probe."""
    shipped = _read(skill_dir / PROBES_FILE).get("vendors") or {}
    local = _read(pathlib.Path(state_dir) / PROBES_FILE).get("vendors") or {} if state_dir else {}
    table = {}
    for vendor in VENDORS:
        row = {"result": NOT_RUN, "where": "never probed"}
        for where, source in (("shipped", shipped), ("this computer", local)):
            record = source.get(vendor)
            if isinstance(record, dict) and record.get("result") in (PASS, FAIL, NOT_RUN):
                if record["result"] != NOT_RUN or row["result"] == NOT_RUN:
                    row = {**record, "where": where}
        table[vendor] = row
    return table


def probe_note(vendor: str, state_dir: Optional[pathlib.Path], skill_dir: pathlib.Path = SKILL_DIR) -> str:
    """The plain fact a join reports about this vendor's probe ("" after a PASS)."""
    row = probe_table(state_dir, skill_dir)[vendor]
    if row["result"] == PASS:
        return ""
    name = VENDORS[vendor][0]
    if row["result"] == FAIL:
        reason = str(row.get("reason") or "see the probe record")
        return (f"Experimental: {name} failed its participant probe ({row['where']}): {reason}; expect "
                "listen-only, or listen with the meeting_audio skill instead.")
    return (f"Experimental: {name} has not been probed with the browser participant yet; if its page "
            "refuses the replaced microphone, the participant only listens and says so.")


def record_probe(state_dir: pathlib.Path, vendor: str, result: str, facts: Dict[str, Any], reason: str = "") -> None:
    path = pathlib.Path(state_dir) / PROBES_FILE
    data = _read(path)
    vendors = data.get("vendors") if isinstance(data.get("vendors"), dict) else {}
    vendors[vendor] = {"result": result, "reason": reason, "facts": facts,
                       "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vendors": vendors}, indent=2), encoding="utf-8")
