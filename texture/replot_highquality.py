"""
Regenerate publication-quality versions of the pipeline visualizations
WITHOUT re-running the oversampling.

It reuses the artifacts that an existing run already saved:
  - <experiment>/logs/pca_scaler.pkl   (fitted StandardScaler + PCA)
  - <experiment>/logs/class_stats.pkl  (centroids, sub-clusters, thresholds)
  - <experiment>/synthetic_samples/    (final synthetic ODF files)
  - data/odf_harmonics                  (original real ODF files)

The original real data and the saved synthetic data are projected through the
saved PCA model, then the existing Visualizer is driven at high DPI and also
exports vector PDFs suitable for a high-impact-factor journal.

Usage examples:
  python scripts/replot_highquality.py
  python scripts/replot_highquality.py --experiment interpolation_v2_target_7500 --dpi 600
  python scripts/replot_highquality.py --max-per-class 0          # use ALL synthetic points
  python scripts/replot_highquality.py --formats png pdf svg
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

# matplotlib is imported before the package so we can configure it globally.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the package importable whether or not it is pip-installed.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PIPELINE_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from odf_pipeline import (  # noqa: E402
    Config,
    DataLoader,
    PCATransformer,
    ClassStatistics,
    Visualizer,
)


def set_publication_style():
    """Apply Matplotlib settings tuned for print-quality journal figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Fonts: serif, embedded as editable TrueType (Type 42) in PDF/PS.
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Slightly heavier lines / larger ticks read better at print scale.
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.markersize": 6,
        "figure.autolayout": False,
    })


def install_savefig_hook(dpi, formats):
    """Make every Visualizer ``plt.savefig('...png')`` emit all requested formats.

    The Visualizer hard-codes PNG paths; we transparently re-route each save to
    the chosen DPI and additionally write vector copies (e.g. PDF/SVG).
    """
    original_savefig = plt.savefig

    def savefig_multi(fname, *args, **kwargs):
        kwargs["dpi"] = dpi
        kwargs.setdefault("bbox_inches", "tight")
        base, _ext = os.path.splitext(str(fname))
        last = None
        for fmt in formats:
            out = f"{base}.{fmt}"
            last = original_savefig(out, *args, **kwargs)
        return last

    plt.savefig = savefig_multi
    return original_savefig


def load_synthetic_pca(synthetic_dir, class_names, pca, max_per_class, seed):
    """Load saved synthetic ODF files per class and project them into PCA space."""
    rng = np.random.default_rng(seed)
    odf_shape_flat = None
    filtered_pca = {}

    for class_name in class_names:
        class_dir = os.path.join(synthetic_dir, class_name)
        files = sorted(glob.glob(os.path.join(class_dir, "*.txt")))

        if not files:
            filtered_pca[class_name] = np.empty((0, pca.pca.n_components_))
            print(f"  {class_name:<40} 0 files (skipped)")
            continue

        if max_per_class and 0 < max_per_class < len(files):
            idx = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]

        rows = []
        for fp in files:
            data = np.loadtxt(fp)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            rows.append(data.flatten())

        X = np.asarray(rows, dtype=np.float64)
        if odf_shape_flat is None:
            odf_shape_flat = X.shape[1]
        filtered_pca[class_name] = pca.transform(X)
        print(f"  {class_name:<40} {len(files):>6} samples -> PCA")

    return filtered_pca


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replot pipeline visualizations at publication quality "
                    "from saved artifacts (no re-sampling)."
    )
    parser.add_argument(
        "--experiment",
        default="interpolation_v2_target_7500",
        help="Experiment folder name under experiments/, or 'default' to use the "
             "top-level logs/synthetic_samples/visualizations folders.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory of original ODF .txt files (default: data/odf_harmonics "
             "relative to the pipeline root).",
    )
    parser.add_argument("--dpi", type=int, default=600,
                        help="Raster DPI for PNG output (default: 600).")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        help="Output formats (default: png pdf). Add svg if needed.")
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="Max synthetic points per class to plot; 0 = use all. "
                             "Subsampling keeps vector files small and scatter readable.")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: <experiment>/visualizations_highres).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic subsampling.")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ paths
    if args.experiment == "default":
        exp_root = PIPELINE_ROOT
        logs_dir = os.path.join(PIPELINE_ROOT, "logs")
        synth_dir = os.path.join(PIPELINE_ROOT, "synthetic_samples")
    else:
        exp_root = os.path.join(PIPELINE_ROOT, "experiments", args.experiment)
        logs_dir = os.path.join(exp_root, "logs")
        synth_dir = os.path.join(exp_root, "synthetic_samples")

    if not os.path.isdir(logs_dir):
        raise SystemExit(f"Logs directory not found: {logs_dir}")
    if not os.path.isdir(synth_dir):
        raise SystemExit(f"Synthetic samples directory not found: {synth_dir}")

    input_dir = args.input_dir or os.path.join(PIPELINE_ROOT, "data", "odf_harmonics")
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Original ODF directory not found: {input_dir}")

    outdir = args.outdir or os.path.join(exp_root, "visualizations_highres")
    os.makedirs(outdir, exist_ok=True)

    # ----------------------------------------------------------------- config
    config = Config()
    config.INPUT_DIR = input_dir
    config.LOGS_DIR = logs_dir
    config.OUTPUT_DIR = synth_dir
    config.VISUALIZATION_DIR = outdir
    config.VIZ_DPI = args.dpi

    print("=" * 80)
    print("HIGH-QUALITY REPLOT (no oversampling)")
    print("=" * 80)
    print(f"  Experiment        : {args.experiment}")
    print(f"  Original ODF dir  : {input_dir}")
    print(f"  Synthetic dir     : {synth_dir}")
    print(f"  Output dir        : {outdir}")
    print(f"  DPI / formats     : {args.dpi} / {', '.join(args.formats)}")
    print(f"  Max per class     : {'ALL' if args.max_per_class == 0 else args.max_per_class}")
    print("=" * 80)

    set_publication_style()
    install_savefig_hook(args.dpi, args.formats)

    t0 = time.time()

    # ----------------------------------------------------- load saved models
    pca = PCATransformer(config)
    pca.load(os.path.join(logs_dir, "pca_scaler.pkl"))

    class_stats = ClassStatistics(config)
    class_stats.load(os.path.join(logs_dir, "class_stats.pkl"))

    # ------------------------------------------------ original data -> PCA
    print("\nLoading original ODF data ...")
    loader = DataLoader(config)
    loader.load_data()
    class_names = loader.class_names_sorted
    X_pca_by_class = {
        c: pca.transform(loader.class_data[c]) for c in class_names
    }
    print(f"  {len(class_names)} classes projected into PCA space.")

    # ----------------------------------------------- synthetic data -> PCA
    print("\nLoading synthetic ODF data ...")
    filtered_pca = load_synthetic_pca(
        synth_dir, class_names, pca, args.max_per_class, args.seed
    )

    # --------------------------------------------------------- regenerate
    print("\nRendering figures ...")
    visualizer = Visualizer(config)
    visualizer.generate_all_visualizations(
        X_pca_by_class, filtered_pca, class_stats, class_names
    )

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print(f"DONE in {dt:.1f}s. High-quality figures written to:\n  {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
