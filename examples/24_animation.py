"""Map animation — a drifting field with a fixed colour scale, saved to GIF."""

import nicefigs as nf
from nicefigs import data


def make(path="24_animation.gif"):
    nf.set_theme("paper")
    lon, lat, frames = data.map_timeseries_cube(n_frames=24)
    fig, _, anim = nf.animate_maps(
        lon,
        lat,
        frames,
        cmap="inferno",
        titles=[f"day {i + 1:02d}" for i in range(len(frames))],
        colorbar_label="intensity",
        fps=8,
    )
    out = nf.save_animation(anim, path, fps=8)
    return out


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    print("wrote", make())
