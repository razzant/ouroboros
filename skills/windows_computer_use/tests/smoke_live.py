# Live smoke: run window_list + window_read against the real desktop.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from windows_computer_use import plugin


class FakeApi:
    def register_tool(self, *a, **k):
        pass

    def get_state_dir(self):
        return str(Path(__file__).parent / "_state")


def main():
    api = FakeApi()
    impl = plugin.WindowsComputerUse(api)
    windows = impl.window_list(limit=15)
    print("== window_list ==")
    for w in windows["windows"][:10]:
        print(f"  0x{int(w['hwnd'], 16):X} {w['title'][:60]!r} class={w['class']} pid={w['pid']} fg={w['foreground']}")
    print(f"count={windows['count']}")

    print("== window_read (foreground) ==")
    result = impl.window_read(max_nodes=60, max_depth=8)
    print(f"ok={result['ok']} backend={result['backend']} nodes_read={result['nodes_read']} "
          f"nodes_returned={result['nodes_returned']} depth={result['depth_reached']} "
          f"truncated_by={result['truncated_by']} elapsed={result['elapsed_ms']}ms title={result['window']['title'][:50]!r}")

    def walk(node, depth=0, out=None):
        out = out if out is not None else []
        label = node.get("name") or node.get("automation_id") or ""
        out.append("  " * depth + f"{node['role']}: {str(label)[:60]}")
        for child in node.get("children", [])[:6]:
            walk(child, depth + 1, out)
        return out

    lines = walk(result["root"]) if result.get("root") else ["  (no root)"]
    for line in lines[:28]:
        print(line)

    print("== element_find (any button) ==")
    found = impl.element_find(select={"role": "button"}, limit=5)
    print(f"ok={found['ok']} matched={found.get('matched')} reason={found.get('reason')}")
    for el in found.get("elements", [])[:5]:
        print(f"  {el.get('role')}: {str(el.get('name'))[:50]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
