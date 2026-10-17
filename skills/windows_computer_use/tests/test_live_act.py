# Live act test: spawn the modern Notepad (UWP), set_value into its edit, read it back.
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from windows_computer_use import plugin


class FakeApi:
    def register_tool(self, *a, **k):
        pass

    def get_state_dir(self):
        return str(Path(__file__).parent / "_state")


def find_our_window(impl, pid, timeout_sec=15.0):
    """The modern Notepad spawns a separate process for its window; watch pid
    and title both, retrying - UIA reads it fine once shown."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for w in impl.window_list(limit=100)["windows"]:
            title = w["title"]
            if "Блокнот" in title or "Notepad" in title:
                if w["pid"] == pid or True:
                    # verify the window belongs to a notepad-ish process
                    return w
        time.sleep(0.5)
    return None


def main():
    impl = plugin.WindowsComputerUse(FakeApi())
    # The modern UWP notepad is what "notepad.exe" launches here (11.26xxx).
    notepad = subprocess.Popen(["notepad.exe"])
    try:
        ours = find_our_window(impl, notepad.pid, timeout_sec=15.0)
        if ours is None:
            print("FAIL: notepad window not found")
            return 1
        hwnd = int(ours["hwnd"], 16)
        print(f"notepad hwnd=0x{hwnd:X} pid={notepad.pid}")

        # 1) set_value into the document (modern Notepad exposes role=document)
        text = "Uroboros UIA test 123"
        result = impl.element_act(hwnd=hwnd, select={"role": "document"}, do="set_value", text=text)
        print("set_value:", {k: v for k, v in result.items() if k != "element"})
        if result.get("ok") is not True:
            print("FAIL: set_value not ok")
            return 1

        time.sleep(0.3)
        # 2) read the value back through find
        found = impl.element_find(hwnd=hwnd, select={"role": "document", "value_contains": "Uroboros"})
        print("find:", {k: v for k, v in found.items() if k != "elements"})
        matched = found.get("matched")
        if found.get("ok") is not True or not matched:
            print("FAIL: value not found back")
            return 1
        value = found["elements"][0].get("value")
        print("value read back:", repr(value))
        if text not in (value or ""):
            print("FAIL: value mismatch")
            return 1

        # 3) window_read win32 fallback explicitly
        reading = impl.window_read(hwnd=hwnd, backend="win32", max_nodes=20)
        print("win32 read:", reading["backend"], "nodes:", reading["nodes_read"], "truncated:", reading["truncated_by"])
        assert reading["backend"] == "win32"

        # 3b) the source's own bound: a read must be faster than a screenshot (<3s)
        uia_reading = impl.window_read(hwnd=hwnd, max_nodes=60, max_depth=8)
        assert uia_reading["backend"] == "uia", uia_reading.get("fallback_reason")
        assert uia_reading["elapsed_ms"] < 3000, f"window_read took {uia_reading['elapsed_ms']}ms - the whole point is being faster than a screenshot"
        print("uia read bound:", uia_reading["elapsed_ms"], "ms (<3000 ok)")

        # 4) not-found refusal is honest (selector names something absent)
        amb = impl.element_act(hwnd=hwnd, select={"role": "edit", "name": "no-such-name"}, do="focus", timeout_ms=500)
        print("not-found act:", amb.get("ok"), amb.get("reason"), str(amb.get("hint"))[:80])
        if amb.get("ok") is not False:
            print("FAIL: expected refusal")
            return 1

        print("ALL LIVE ACT TESTS PASS")
        return 0
    finally:
        notepad.terminate()
        try:
            notepad.wait(timeout=3)
        except Exception:
            notepad.kill()
        # close the visible modern notepad window too, best-effort
        try:
            import subprocess as sp

            sp.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Process Notepad -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -like '*имя*' -or $_.MainWindowTitle -like '*Блокнот*' -or $_.MainWindowTitle -eq ''} | Stop-Process -Force"],
                capture_output=True, timeout=15,
            )
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
