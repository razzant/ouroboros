"""Written settings addresses must name a page that survives the Settings rework."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LIVING = (
    "Models", "Agents", "Your rules", "Permissions and checks", "Accounts and keys",
    "Tools", "Appearance", "Notifications", "Voice and calls", "This computer",
)
PENDING = {}
# A release's row of README's Version History says what that release called a page; history is not rewritten, and
# every release moves the rows, so they are known by their shape and not by a line number.
RELEASE_ROW = re.compile(r"^\| \d+\.\d+\.\d+\S* \| \d{4}-\d{2}-\d{2} \|")
# Exact (file, line) exceptions only: blocked bundled-skill releases.
# Each entry must explain the debt; new addresses elsewhere in these files still fail.
PROTECTED_KNOWN = {
    ('skills/telegram/lib/telegram_api.py', 1002): "wording only; changes at the skill's next release",
    ('skills/telegram/lib/telegram_api.py', 1007): "wording only; changes at the skill's next release",
    ('skills/telegram/lib/telegram_connection.py', 1): "wording only; changes at the skill's next release",
    ('skills/telegram/lib/telegram_connection.py', 137): "wording only; changes at the skill's next release",
    ('skills/telegram/lib/telegram_connection.py', 190): "wording only; changes at the skill's next release",
    ('skills/telegram/lib/telegram_connection.py', 211): "wording only; changes at the skill's next release",
    ('skills/telegram/lib/telegram_token.py', 210): "wording only; changes at the skill's next release",
    ('skills/telegram/plugin.py', 1451): "wording only; changes at the skill's next release",
    ('skills/telegram/scripts/companion.py', 496): "wording only; changes at the skill's next release",
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
    assert RELEASE_ROW.match("| 7.9.0 | 2026-10-28 | **feat: Settings → Accounts holds several API keys.**")
    assert not RELEASE_ROW.match("Add the key in Settings → Secrets.")
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
        text = (ROOT / name).read_text(encoding="utf-8")
        rows = text.splitlines()
        for line, address in dead_addresses(text):
            if name == "README.md" and RELEASE_ROW.match(rows[line - 1]):
                continue
            location = (name, line)
            if location in PROTECTED_KNOWN:
                seen.add(location)
            else:
                failures.append(f"{name}:{line}: {address}")
    assert seen == set(PROTECTED_KNOWN), "Remove stale protected-address exceptions"
    assert not failures, "Dead Settings addresses:\n" + "\n".join(failures)
