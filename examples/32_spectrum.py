"""Eigen-spectra — eigenvalue decay and cumulative explained variance, side by side."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    spectra = data.eigen_spectra()
    fig, (ax1, ax2) = nf.subplots(1, 2, width="full", aspect=0.42, layout="constrained")
    nf.spectrum(spectra, ax=ax1, reference_alpha=1.0, title="Eigenvalue decay")
    nf.spectrum(spectra, ax=ax2, cumulative=True, legend=False, title="Cumulative explained variance")
    nf.label_panels([ax1, ax2], use_ax_title=True)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "32_spectrum", formats=("png", "pdf"))
