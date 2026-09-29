"""
Standalone, publication-quality journal figures for the log-normal grain-size
parameters (mu, sigma of ln(ESD)), mirroring the GSH/PCA journal figure set:

  1. journal_musigma_combined.{png,pdf}          - all classes overlaid
  2. journal_musigma_class_<CLASS>.{png,pdf}      - one per highlighted class

Experimental (per-condition fitted) parameters are drawn as large coloured
stars; the oversampled parameters are small coloured dots, coloured by material
class. Colours match the GSH journal figures (same sorted-class -> palette map).

Nothing is re-sampled; everything is read from the saved stats files:
  experimental : <stats_dir>/<name>_stats.txt            (mu, sigma = first 2 #s)
  oversampled  : <sampled_dir>/<class>/<class>_<n>_stats.txt

Usage:
  python journal_mu_sigma_figures.py                       # combined figure
  python journal_mu_sigma_figures.py --highlight AZ31 ME21 # per-class figures
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Live data locations (the original plot script's paths are stale).
DEFAULT_STATS = "data/stats"
DEFAULT_SAMPLED = (
    "data/stats_combined_sampled"
)
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, "journal_musigma")

# Axis labels: location / scale of the log-normal ESD distribution.
MU_LABEL = r"$\mu_{\ln(\mathrm{ESD})}$"
SIGMA_LABEL = r"$\sigma_{\ln(\mathrm{ESD})}$"


# ----------------------------------------------------------------------------
# Style (identical to the GSH journal figures for a consistent look)
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


# Curated vivid qualitative palette (matches the GSH figures, sorted-class order).
_DISTINCT = [
    "#e6194B",  # 0  crimson      (AZ31_extruded)
    "#3cb44b",  # 1  green        (AZ31_extruded_heattreated)
    "#ffd700",  # 2  gold         (ME21_extruded)
    "#f58231",  # 3  orange       (Mg-10Gd-0.5Mn_extruded)
    "#911eb4",  # 4  purple       (Mg-10Gd-1Mn_extruded)
    "#42d4f4",  # 5  cyan         (Mg-10Gd_extruded)
    "#f032e6",  # 6  magenta      (Mg-2Gd-0.5Mn_extruded)
    "#9A6324",  # 7  brown        (Mg-2Gd-1Mn_extruded)
    "#1f9e89",  # 8  teal         (Mg-2Gd_extruded)
    "#800000",  # 9  maroon       (Mg-5Gd-0.5Mn_extruded)
    "#808000",  # 10 olive        (Mg-5Gd-1Mn_extruded)
    "#000075",  # 11 navy         (Mg-5Gd_extruded)
    "#ff6fb5",  # 12 rose         (Z1_extruded)
    "#4363d8",  # 13 blue         (ZNd10_extruded)
    "#c71585",  # 14 medium violet red (ZNd10_extruded_heattreated)
    "#ff4500",  # 15 orange-red   (ZX10_extruded)
    "#6a3d9a",  # 16 deep violet  (ZX10_extruded_heattreated)
]


def get_class_colors(n):
    if n <= len(_DISTINCT):
        from matplotlib.colors import to_rgba
        return np.array([to_rgba(c) for c in _DISTINCT[:n]])
    return plt.cm.gist_ncar(np.linspace(0, 1, n))


def prettify(name):
    """Human-friendly class label ('extruded' is dropped as it is common)."""
    return name.replace("_extruded", "").replace("_", " ").strip()


def select_lively(P, seed, grid=52, keep_percentile=82):
    """Spatially stratified thinning so a cluster reads as lively and even.

    A uniform subsample preserves density, leaving sparse edges dead and dense
    cores solid. Instead we cap how many points each grid cell may keep: sparse
    cells keep everything (periphery stays populated) while only crowded cores
    are thinned.
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
    kept, seen = [], {}
    for idx in rng.permutation(n):
        c = int(cell[idx])
        k = seen.get(c, 0)
        if k < cap:
            kept.append(idx)
            seen[c] = k + 1
    return np.array(sorted(kept), dtype=int)


def save_all(fig, base, dpi, formats):
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=dpi)
    print(f"  saved {os.path.basename(base)}.{{{','.join(formats)}}}")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _read_mu_sigma(path):
    """Return (mu_lnESD, sigma_lnESD) = the first two numeric values, or None."""
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for tok in s.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
                if len(vals) >= 2:
                    return vals[0], vals[1]
    return None


def _group_key(base):
    """Map a stats filename to its material-class key."""
    if "extruded_heattreated" in base:
        i = base.find("extruded_heattreated")
        return base[: i + len("extruded_heattreated")]
    if "extruded" in base:
        i = base.find("extruded")
        return base[: i + len("extruded")]
    return base.split(".")[0].rstrip("_")


def collect_original(stats_dir):
    """dict[class] -> (N, 2) array of experimental (mu, sigma)."""
    groups = {}
    for path in glob.glob(os.path.join(stats_dir, "*_stats.txt")):
        pair = _read_mu_sigma(path)
        if pair:
            groups.setdefault(_group_key(os.path.basename(path)), []).append(pair)
    return {k: np.asarray(v, float) for k, v in groups.items()}


def collect_sampled(sampled_dir, max_per_class, seed):
    """dict[class] -> (N, 2) array of oversampled (mu, sigma)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for material in sorted(os.listdir(sampled_dir)):
        mat_dir = os.path.join(sampled_dir, material)
        if not os.path.isdir(mat_dir):
            continue
        rgx = re.compile(rf"^{re.escape(material)}_\d+_stats\.txt$")
        files = sorted(f for f in os.listdir(mat_dir) if rgx.match(f))
        if not files:
            continue
        if max_per_class and 0 < max_per_class < len(files):
            idx = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        pairs = []
        for fn in files:
            pr = _read_mu_sigma(os.path.join(mat_dir, fn))
            if pr:
                pairs.append(pr)
        if pairs:
            groups[material] = np.asarray(pairs, float)
    return groups


def resolve_class(query, class_names):
    """Resolve a short query (e.g. 'AZ31', 'ME21') to a full class name."""
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


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _add_legends(fig, ax, class_names, colors, ncol=1):
    """Two legends: marker meaning (top) and per-class colours (below)."""
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


def make_combined(orig, sampled, class_names, colors, outbase, dpi, formats,
                  seed=42):
    fig, ax = plt.subplots(figsize=(17.5, 10.5))

    # 1) Synthetic cloud (light, stratified for an even, lively spread).
    cloud_pts = []
    for i, c in enumerate(class_names):
        S = sampled.get(c)
        if S is None or not len(S):
            continue
        col = colors[i]
        sel = select_lively(S[:, :2], seed=seed + i, grid=46,
                            keep_percentile=58)
        ax.scatter(S[sel, 0], S[sel, 1], color=col, s=24, alpha=0.22,
                   edgecolors="none", zorder=1, rasterized=True)
        cloud_pts.append(S[sel, :2])

    # 2) Experimental stars with a white halo so they pop on any background.
    star_pts = []
    for i, c in enumerate(class_names):
        R = orig.get(c)
        if R is None or not len(R):
            continue
        ax.scatter(R[:, 0], R[:, 1], s=760, color="white", marker="*",
                   edgecolors="none", zorder=4)
        ax.scatter(R[:, 0], R[:, 1], s=600, color=colors[i], marker="*",
                   edgecolors="black", linewidths=1.1, zorder=5)
        star_pts.append(R[:, :2])

    # Zoom in: clip the axes to the dense dot cloud (robust percentiles drop
    # sparse outliers / whitespace) while keeping every experimental star in.
    if cloud_pts:
        P = np.vstack(cloud_pts)
        xlo, xhi = np.percentile(P[:, 0], [0.5, 99.5])
        ylo, yhi = np.percentile(P[:, 1], [0.5, 99.5])
        if star_pts:
            Rall = np.vstack(star_pts)
            xlo, xhi = min(xlo, Rall[:, 0].min()), max(xhi, Rall[:, 0].max())
            ylo, yhi = min(ylo, Rall[:, 1].min()), max(yhi, Rall[:, 1].max())
        px = (xhi - xlo) * 0.02
        py = (yhi - ylo) * 0.02
        ax.set_xlim(xlo - px, xhi + px)
        ax.set_ylim(ylo - py, yhi + py)

    ax.set_xlabel(MU_LABEL, fontsize=30)
    ax.set_ylabel(SIGMA_LABEL, fontsize=30)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.tick_params(width=1.3, length=6, labelsize=24)

    _add_legends(fig, ax, class_names, colors)
    fig.tight_layout(rect=(0, 0, 0.70, 1))
    save_all(fig, outbase, dpi, formats)
    plt.close(fig)


def make_highlight(target, orig, sampled, class_names, colors, outbase, dpi,
                   formats, seed=42):
    ti = class_names.index(target)
    tcol = colors[ti]

    fig, ax = plt.subplots(figsize=(7.6, 7.0))

    S = sampled.get(target)
    if S is not None and len(S):
        sel = select_lively(S[:, :2], seed=seed + ti)
        ax.scatter(S[sel, 0], S[sel, 1], color=tcol, s=34, alpha=0.55,
                   edgecolors="none", zorder=2, rasterized=True)
    R = orig.get(target, np.empty((0, 2)))
    if len(R):
        ax.scatter(R[:, 0], R[:, 1], color=tcol, s=720, alpha=1.0,
                   marker="*", edgecolors="black", linewidths=1.6, zorder=4)

    # Frame the class extent (real + synthetic) tightly.
    pts = [R] if len(R) else []
    if S is not None and len(S):
        pts.append(S[:, :2])
    if pts:
        P = np.vstack(pts)
        xmin, ymin = P[:, 0].min(), P[:, 1].min()
        xmax, ymax = P[:, 0].max(), P[:, 1].max()
        px = max((xmax - xmin) * 0.08, 0.02)
        py = max((ymax - ymin) * 0.08, 0.01)
        ax.set_xlim(xmin - px, xmax + px)
        ax.set_ylim(ymin - py, ymax + py)

    ax.set_xlabel(MU_LABEL)
    ax.set_ylabel(SIGMA_LABEL)
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


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Standalone journal figures for mu/sigma of log-normal ESD.")
    p.add_argument("--stats-dir", default=DEFAULT_STATS)
    p.add_argument("--sampled-dir", default=DEFAULT_SAMPLED)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--max-per-class", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--highlight", nargs="+", default=None,
                   help="Generate single-class figures for these classes "
                        "(e.g. AZ31 ME21) instead of the combined figure.")
    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    set_style()

    print("Reading experimental stats ...")
    orig = collect_original(args.stats_dir)
    print(f"  {sum(len(v) for v in orig.values())} points / {len(orig)} classes")

    print("Reading oversampled stats ...")
    sampled = collect_sampled(args.sampled_dir, args.max_per_class, args.seed)
    print(f"  {sum(len(v) for v in sampled.values())} points / {len(sampled)} classes")

    class_names = sorted(set(orig) | set(sampled))
    colors = get_class_colors(len(class_names))

    if args.highlight:
        for query in args.highlight:
            target = resolve_class(query, class_names)
            print(f"Rendering highlight figure for '{target}' ...")
            make_highlight(
                target, orig, sampled, class_names, colors,
                os.path.join(args.outdir, f"journal_musigma_class_{target}"),
                args.dpi, args.formats, seed=args.seed)
        print(f"\nDone. Figures in:\n  {args.outdir}")
        return

    print("Rendering combined figure ...")
    make_combined(orig, sampled, class_names, colors,
                  os.path.join(args.outdir, "journal_musigma_combined"),
                  args.dpi, args.formats, seed=args.seed)
    print(f"\nDone. Figures in:\n  {args.outdir}")


if __name__ == "__main__":
    main()
