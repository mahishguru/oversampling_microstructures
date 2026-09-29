"""
Generate two standalone, publication-quality figures from the latest run:

  1. journal_combined_2d.{png,pdf}  - PC1 vs PC2
  2. journal_combined_3d.{png,pdf}  - PC1 vs PC2 vs PC3

Both overlay the experimental (real) ODFs as large coloured stars and the
oversampled (synthetic) ODFs as small coloured dots, coloured by material
class. No oversampling is re-run; everything is rebuilt from saved artifacts.

Usage:
  python scripts/journal_figures.py
  python scripts/journal_figures.py --max-per-class 0      # use ALL synthetic points
  python scripts/journal_figures.py --dpi 600 --azim -60 --elev 22
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PIPELINE_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Reuse data-loading helpers from the high-quality replot script.
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "texture"))
from replot_highquality import load_synthetic_pca  # noqa: E402
from odf_pipeline import Config, DataLoader, PCATransformer  # noqa: E402


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.3,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
    })


# Curated vivid qualitative palette (no greys / no near-white pales) so all 17
# material classes are separable and the figures read as lively.
_DISTINCT = [
    "#e6194B",  # 0  crimson
    "#3cb44b",  # 1  green
    "#ffd700",  # 2  gold        (ME21_extruded)
    "#f58231",  # 3  orange
    "#911eb4",  # 4  purple
    "#42d4f4",  # 5  cyan
    "#f032e6",  # 6  magenta
    "#9A6324",  # 7  brown
    "#1f9e89",  # 8  teal
    "#800000",  # 9  maroon
    "#808000",  # 10 olive
    "#000075",  # 11 navy
    "#ff6fb5",  # 12 rose
    "#4363d8",  # 13 blue        (ZNd10_extruded)
    "#c71585",  # 14 medium violet red
    "#ff4500",  # 15 orange-red  (ZX10_extruded)
    "#6a3d9a",  # 16 deep violet (ZX10_extruded_heattreated)
]


def get_class_colors(n):
    """Return n visually distinct colours."""
    if n <= len(_DISTINCT):
        from matplotlib.colors import to_rgba
        return np.array([to_rgba(c) for c in _DISTINCT[:n]])
    return plt.cm.gist_ncar(np.linspace(0, 1, n))


def select_lively(P, seed, grid=52, keep_percentile=82):
    """Spatially stratified thinning so a cluster reads as lively and even.

    A uniform random subsample preserves density, so dense cores stay solid
    while sparse edges look dead. Instead, we lay a regular grid over the data
    extent and cap how many points each occupied cell may keep: sparse cells
    keep all of their points (the periphery stays populated), while only the
    most crowded cores are thinned. ``keep_percentile`` sets the cap from the
    occupied-cell density distribution (higher -> keep more / denser).
    """
    n = len(P)
    if n == 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    xmin, ymin = P[:, 0].min(), P[:, 1].min()
    xmax, ymax = P[:, 0].max(), P[:, 1].max()
    ex = (xmax - xmin) or 1.0
    ey = (ymax - ymin) or 1.0
    gx = np.clip(((P[:, 0] - xmin) / ex * grid).astype(int), 0, grid - 1)
    gy = np.clip(((P[:, 1] - ymin) / ey * grid).astype(int), 0, grid - 1)
    cell = gx * grid + gy
    counts = np.bincount(cell)
    occupied = counts[counts > 0]
    cap = max(int(np.percentile(occupied, keep_percentile)), 1)
    kept = []
    seen = {}
    for idx in rng.permutation(n):
        c = int(cell[idx])
        k = seen.get(c, 0)
        if k < cap:
            kept.append(idx)
            seen[c] = k + 1
    return np.array(sorted(kept), dtype=int)


def prettify(name):
    """Human-friendly class label ('extruded' is dropped as it is common to all)."""
    return name.replace("_extruded", "").replace("_", " ").strip()


# Axis labels: make clear the axes are principal components of the GSH
# (generalised spherical harmonic) ODF coefficients.
PC_X_LABEL = "PC 1 of GSH coefficients"
PC_Y_LABEL = "PC 2 of GSH coefficients"
PC_Z_LABEL = "PC 3 of GSH coefficients"


def save_all(fig, base, dpi, formats):
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=dpi)
    print(f"  saved {os.path.basename(base)}.{{{','.join(formats)}}}")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_2d(X_real, X_synth, class_names, colors, outbase, dpi, formats,
            seed=42):
    # Page-width landscape: plot on the left, legends in a wide right gutter so
    # the per-class legend text can stay large.
    fig, ax = plt.subplots(figsize=(17.5, 10.5))

    for i, c in enumerate(class_names):
        col = colors[i]
        S = X_synth.get(c, np.empty((0, 2)))
        if len(S):
            sel = select_lively(S[:, :2], seed=seed + i)
            ax.scatter(S[sel, 0], S[sel, 1], color=col, s=26, alpha=0.6,
                       edgecolors="none", zorder=1, rasterized=True)
    for i, c in enumerate(class_names):
        col = colors[i]
        R = X_real[c]
        ax.scatter(R[:, 0], R[:, 1], color=col, s=600, alpha=1.0,
                   marker="*", edgecolors="black", linewidths=1.4, zorder=3)

    ax.set_xlabel(PC_X_LABEL)
    ax.set_ylabel(PC_Y_LABEL)
    ax.margins(0.03)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.tick_params(width=1.3, length=6)

    _add_legends(fig, ax, class_names, colors, ncol=1, anchor=(1.02, 1.0))
    fig.tight_layout(rect=(0, 0, 0.70, 1))
    save_all(fig, outbase, dpi, formats)
    plt.close(fig)


def make_3d(X_real, X_synth, class_names, colors, outbase, dpi, formats,
            azim, elev):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection="3d")

    for i, c in enumerate(class_names):
        col = colors[i]
        S = X_synth.get(c, np.empty((0, 3)))
        if len(S):
            ax.scatter(S[:, 0], S[:, 1], S[:, 2], color=col, s=12, alpha=0.40,
                       edgecolors="none", depthshade=False, zorder=1,
                       rasterized=True)
    for i, c in enumerate(class_names):
        col = colors[i]
        R = X_real[c]
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], color=col, s=300, alpha=1.0,
                   marker="*", edgecolors="black", linewidths=1.0,
                   depthshade=False, zorder=3)

    ax.set_xlabel(PC_X_LABEL, labelpad=22)
    ax.set_ylabel(PC_Y_LABEL, labelpad=22)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(PC_Z_LABEL, labelpad=26, rotation=90)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis="z", pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    # Enlarge the 3D content to fill the panel (reduce surrounding whitespace).
    try:
        ax.set_box_aspect(None, zoom=1.12)
    except TypeError:
        pass

    _add_legends(fig, ax, class_names, colors, ncol=1, anchor=(1.0, 0.98),
                 is_3d=True)
    fig.subplots_adjust(left=0.0, right=0.74, bottom=0.08, top=0.99)
    save_all(fig, outbase, dpi, formats)
    plt.close(fig)


def resolve_class(query, class_names):
    """Resolve a short query (e.g. 'ZX10', 'Mg-5Gd') to a full class name.

    Preference order: exact match, then the shortest name matching the
    '<query>_' prefix (selects the base variant over heat-treated/alloyed ones),
    then any substring match.
    """
    if query in class_names:
        return query
    prefixed = sorted((c for c in class_names if c.startswith(query + "_")),
                      key=len)
    if prefixed:
        return prefixed[0]
    subs = sorted((c for c in class_names if query in c), key=len)
    if subs:
        return subs[0]
    raise SystemExit(
        f"No class matches '{query}'. Available: {', '.join(class_names)}")


def make_class_highlight(target, X_real, X_synth, class_names, colors,
                         outbase, dpi, formats):
    """Single-class figure: only this class's real stars + synthetic dots.

    The class colour matches the one used in the combined journal figures.
    """
    ti = class_names.index(target)
    tcol = colors[ti]

    # Near-square, tight: two of these sit side-by-side across a page width.
    fig, ax = plt.subplots(figsize=(7.6, 7.0))

    # Oversampled (synthetic) dots.
    St = X_synth.get(target, np.empty((0, 2)))
    if len(St):
        ax.scatter(St[:, 0], St[:, 1], color=tcol, s=40, alpha=0.60,
                   edgecolors="none", zorder=2, rasterized=True)
    # Experimental (real) stars.
    Rt = X_real[target]
    ax.scatter(Rt[:, 0], Rt[:, 1], color=tcol, s=720, alpha=1.0,
               marker="*", edgecolors="black", linewidths=1.6, zorder=4)

    # Frame the class extent (real + synthetic) tightly.
    pts = [Rt]
    if len(St):
        pts.append(St)
    P = np.vstack(pts)
    xmin, ymin = P[:, 0].min(), P[:, 1].min()
    xmax, ymax = P[:, 0].max(), P[:, 1].max()
    px = max((xmax - xmin) * 0.08, 0.6)
    py = max((ymax - ymin) * 0.08, 0.6)
    ax.set_xlim(xmin - px, xmax + px)
    ax.set_ylim(ymin - py, ymax + py)

    ax.set_xlabel(PC_X_LABEL)
    ax.set_ylabel(PC_Y_LABEL)
    ax.set_title(prettify(target), fontsize=27, fontweight="bold", pad=12)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.tick_params(width=1.3, length=6)

    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=tcol,
               markeredgecolor="black", markersize=34,
               label="Experimental (real)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=tcol,
               markeredgecolor="none", markersize=22,
               label="Oversampled (synthetic)"),
    ]
    leg = ax.legend(handles=handles, loc="best", frameon=True,
                    fontsize=21, borderaxespad=0.6, handletextpad=0.5,
                    labelspacing=0.5, borderpad=0.6)
    leg.get_frame().set_edgecolor("0.5")

    fig.tight_layout(pad=0.6)
    save_all(fig, outbase, dpi, formats)
    plt.close(fig)


def _add_legends(fig, ax, class_names, colors, ncol, anchor, is_3d=False):
    """Two legends: marker meaning (top) and per-class colours (below)."""
    # Marker-type legend (shape meaning, neutral grey).
    marker_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="0.35",
               markeredgecolor="black", markersize=30,
               label="Experimental (real)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.55",
               markeredgecolor="none", markersize=18,
               label="Oversampled (synthetic)"),
    ]
    leg1 = fig.legend(handles=marker_handles, loc="upper left",
                      bbox_to_anchor=(0.705, 0.99), frameon=True,
                      title="Marker", fontsize=23, title_fontsize=26,
                      borderaxespad=0.0)
    leg1.get_frame().set_edgecolor("0.5")

    # Per-class colour legend (17 entries -> kept large in a wide gutter).
    class_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[i],
               markeredgecolor="none", markersize=22, label=prettify(c))
        for i, c in enumerate(class_names)
    ]
    leg2 = fig.legend(handles=class_handles, loc="upper left",
                      bbox_to_anchor=(0.705, 0.84), frameon=True,
                      title="Material class", ncol=ncol, fontsize=22,
                      title_fontsize=26, borderaxespad=0.0,
                      labelspacing=0.28, handletextpad=0.4)
    leg2.get_frame().set_edgecolor("0.5")


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Standalone 2D/3D journal figures.")
    p.add_argument("--experiment", default="interpolation_v2_target_7500")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--max-per-class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--azim", type=float, default=-66.0)
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--outdir", default=None)
    p.add_argument("--highlight", nargs="+", default=None,
                   help="Generate single-class highlight figures for these "
                        "classes (e.g. ZX10 Mg-5Gd) instead of the combined "
                        "2D/3D figures.")
    args = p.parse_args(argv)

    exp_root = os.path.join(PIPELINE_ROOT, "experiments", args.experiment)
    logs_dir = os.path.join(exp_root, "logs")
    synth_dir = os.path.join(exp_root, "synthetic_samples")
    input_dir = os.path.abspath(
        args.input_dir or os.path.join(PIPELINE_ROOT, "data", "odf_harmonics"))
    outdir = args.outdir or os.path.join(exp_root, "visualizations_highres")
    os.makedirs(outdir, exist_ok=True)

    config = Config()
    config.INPUT_DIR = input_dir
    config.LOGS_DIR = logs_dir

    set_style()

    print("Loading PCA model and data ...")
    pca = PCATransformer(config)
    pca.load(os.path.join(logs_dir, "pca_scaler.pkl"))

    loader = DataLoader(config)
    loader.load_data()
    class_names = loader.class_names_sorted
    X_real = {c: pca.transform(loader.class_data[c]) for c in class_names}

    print("Loading synthetic samples ...")
    X_synth = load_synthetic_pca(synth_dir, class_names, pca,
                                 args.max_per_class, args.seed)

    colors = get_class_colors(len(class_names))

    if args.highlight:
        for query in args.highlight:
            target = resolve_class(query, class_names)
            print(f"Rendering highlight figure for '{target}' ...")
            make_class_highlight(
                target, X_real, X_synth, class_names, colors,
                os.path.join(outdir, f"journal_class_{target}"),
                args.dpi, args.formats)
        print(f"\nDone. Figures in:\n  {outdir}")
        return

    print("Rendering 2D figure ...")
    make_2d(X_real, X_synth, class_names, colors,
            os.path.join(outdir, "journal_combined_2d"),
            args.dpi, args.formats, seed=args.seed)

    print(f"\nDone. Figures in:\n  {outdir}")


if __name__ == "__main__":
    main()
