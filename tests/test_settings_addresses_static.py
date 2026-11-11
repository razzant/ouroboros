"""Written settings addresses must name a page that survives the Settings rework."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LIVING = (
    "Models", "Agents", "Your rules", "Permissions and checks", "Accounts and keys",
    "Tools", "Appearance", "Notifications", "Voice and calls", "This computer",
)
# The convener empties this list on the integrated tree; each owner rewrites its files.
PENDING = {
    "web/modules/": "B2/C1/C2/D/E1/E2/E3: controls, shell, pages, dashboard and editors",
    "docs/DESIGN.md": "Books and gates: convener",
    "docs/architecture/03-web-ui-pages-and-buttons.md": "Books and gates: convener",
    "docs/development/": "Books and gates: convener",
}
# Exact (file, line) exceptions only: protected text and blocked bundled-skill releases.
# Each entry must explain the debt; new addresses elsewhere in these files still fail.
PROTECTED_KNOWN = {
    ('README.md', 502): 'COMMON: release history is protected',
    ('README.md', 505): 'COMMON: release history is protected',
    ('README.md', 506): 'COMMON: release history is protected',
    ('skills/live_conversation/SKILL.md', 22): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/live_conversation/plugin.py', 109): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/live_conversation/ui/call.js', 140): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_api.py', 1002): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_api.py', 1007): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_connection.py', 1): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_connection.py', 137): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_connection.py', 190): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_connection.py', 211): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/lib/telegram_token.py', 210): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/plugin.py', 1451): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
    ('skills/telegram/scripts/companion.py', 496): 'WO G.4: delivering changed bundled bytes requires a skill version bump',
}
ADDRESS = re.compile(r"\bSettings\s*(?:→|›|->|>)\s*(?=[A-Za-z])")
PAGE = re.compile(r"(?:" + "|".join(re.escape(page) for page in LIVING) + r")(?![\w-])")


def dead_addresses(text):
    # Join wrapped Python literals while retaining newlines for source line numbers.
    text = re.sub(r'''["']\s*\n\s*[furb]*["']''',
                  lambda match: "\n" * match.group().count("\n"), text)
    for match in ADDRESS.finditer(text):
        # macOS System Settings is a different product, not an Ouroboros address.
        if re.search(r"\bSystem\s+$", text[:match.start()]):
            continue
        if not PAGE.match(text, match.end()):
            line = text.count("\n", 0, match.start()) + 1
            yield line, text[match.start():match.end() + 90].splitlines()[0]


def test_written_settings_addresses_name_living_pages():
    broken = "Settings→Behavior; Settings › Advanced; Settings > Secrets; Settings->Providers; Settings → Accounts; Settings › Review; Settings > Runtime Limits"
    living = "Settings→Models; Settings › Agents; Settings > Your rules; Settings->Permissions and checks; Settings → Accounts and keys; Settings › Tools; Settings > Appearance; Settings->Notifications; Settings → Voice and calls; Settings › This computer › Advanced; System Settings → Privacy & Security"
    assert len(list(dead_addresses(broken))) == 7
    assert list(dead_addresses(living)) == []
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "ouroboros/", "supervisor/", "skills/",
         "web/modules/", "docs/", "prompts/", "README.md"], cwd=ROOT,
    ).decode("utf-8").split("\0")
    failures, seen = [], set()
    for name in tracked:
        if Path(name).suffix not in {".py", ".js", ".md"}:
            continue
        if any(name.startswith(path) if path.endswith("/") else name == path for path in PENDING):
            continue
        for line, address in dead_addresses((ROOT / name).read_text(encoding="utf-8")):
            location = (name, line)
            if location in PROTECTED_KNOWN:
                seen.add(location)
            else:
                failures.append(f"{name}:{line}: {address}")
    assert seen == set(PROTECTED_KNOWN), "Remove stale protected-address exceptions"
    assert not failures, "Dead Settings addresses:\n" + "\n".join(failures)
