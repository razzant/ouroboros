"""The skill-review rubric follows both supported widget entry contracts."""
import pathlib
import re


def test_every_seat_reads_both_module_entry_modes_and_the_unchanged_frame():
    """The actual rubric admits captured ESM files without weakening the opaque frame."""
    from ouroboros.tools.review_skill import skill_rubric

    rows = dict(re.findall(r"^\| \d+ \| ([a-z_]+) \| (.*)$", skill_rubric(), flags=re.M))
    widget, namespace = rows["widget_module_safety"], rows["extension_namespace_discipline"]
    frame = (pathlib.Path(__file__).resolve().parents[1] / "web/modules/widget_module.js").read_text(encoding="utf-8")
    sandbox, allow = (re.search(rf"export const {name} = '([^']+)';", frame).group(1)
                      for name in ("WIDGET_FRAME_SANDBOX", "WIDGET_FRAME_ALLOW"))
    assert f'sandbox="{sandbox}" allow="{allow}"' in widget and "no ``allow-same-origin``" in widget
    assert "classic entry" in widget, "the inlined /module/<entry> source is only the classic default"
    assert widget.index("classic entry") < widget.index("GET /api/extensions/<skill>/module/<entry>")
    assert 'explicit ``render.entry_type: "module"``' in widget and "declared ``render.assets``" in widget
    assert "captured at publication" in widget and "GET/HEAD" in widget
    assert "declared files" in widget[widget.index("(a)"):widget.index("(b)")]
    assert "import.meta.url" in widget[widget.index("Acceptable interactions"):]
    assert "declared files" in namespace[namespace.index("media sourced from"):].split("?", 1)[0]
