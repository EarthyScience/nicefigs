"""Schematic: a small pipeline drawn at its printed size from boxes, arrows, discs, bands and glyphs."""

import nicefigs as nf


def make():
    nf.set_theme(nf.Theme.from_base(7.2))
    fig, ax = nf.schematic_axes(5.5, 1.9)
    th = nf.get_theme()
    blue, orange = nf.NICE_COLORS["blue"], nf.NICE_COLORS["orange"]
    nf.group_band(ax, (1.7, 0.1), 2.4, 1.7, "run chunk by chunk")
    steps = [
        (0.8, "read", "target and prediction"),
        (2.9, "compare", "view, then metric"),
        (4.7, "store", "one folder each"),
    ]
    for k, (x, name, detail) in enumerate(steps):
        fill = nf.tint(blue, 0.2) if k == 1 else "white"
        nf.diagram_box(
            ax, (x, 1.2), 1.3, 0.5, facecolor=fill, edgecolor=th.text_color, linewidth=0.7, rounding=0.03
        )
        nf.label_pair(ax, (x, 1.2), name, detail, bold=True)
        nf.numbered_disc(ax, (x - 0.65, 1.45), k + 1)
    for x0, x1 in ((1.45, 2.25), (3.55, 4.05)):
        nf.diagram_arrow(ax, (x0, 1.2), (x1, 1.2), style="line", width=0.7, head_length=3.4, head_width=3.0)
    nf.grid_cells(ax, (0.3, 0.2), 5, 3, 0.15)
    nf.grid_cells(ax, (2.15, 0.2), 5, 3, 0.15, fills={(2, 1): nf.tint(orange, 0.5)})
    for k in range(3):
        nf.folder_icon(ax, (4.2 + k * 0.3, 0.3), 0.24, 0.18)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "35_schematic", formats=("png", "pdf"), check_text=True)
