#!/usr/bin/env python3
"""Calibrate grain-count prediction formula against actual DREAM3D outputs.

Reads .dream3d (HDF5) files from the backup RVE directory and compares
actual grain counts with formula predictions based on (mu, sigma) from
the corresponding stats files.

Output: prints a table and the calibration factor k such that:
    n_grains ≈ k * V_RVE / ((pi/6) * exp(3*mu + 4.5*sigma^2))
"""

import os
import math
import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("h5py is required: pip install h5py")

# ─── Configuration ───────────────────────────────────────────────────────────
RVE_ROOT = "data/rve"
STATS_ROOT = "data/stats_combined_sampled"

# RVE geometry from template.json
NX, NY, NZ = 300, 300, 1
DX, DY, DZ = 2.0, 2.0, 2.0  # µm per voxel
VOLUME = NX * DX * NY * DY * NZ * DZ  # 720,000 µm³

# Materials and sample IDs to check
MATERIALS = [
    "AZ31_extruded",
    "ME21_extruded",
    "Mg-5Gd_extruded",
    "Mg-10Gd_extruded",
    "Mg-2Gd_extruded",
    "Z1_extruded",
    "ZX10_extruded",
    "ZNd10_extruded",
]
SAMPLE_IDS = [1, 5, 10, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_grain_count(dream3d_path: str) -> int | None:
    """Extract grain count from .dream3d HDF5 file."""
    try:
        with h5py.File(dream3d_path, "r") as f:
            base = "DataContainers/SyntheticVolumeDataContainer"
            # Method 1: CellFeatureData array length - 1
            feature_data = f"{base}/CellFeatureData"
            if feature_data in f:
                for key in f[feature_data].keys():
                    shape = f[feature_data][key].shape
                    return int(shape[0] - 1)
            # Method 2: max(FeatureIds)
            cell_data = f"{base}/CellData"
            if cell_data in f and "FeatureIds" in f[cell_data]:
                return int(np.max(f[cell_data]["FeatureIds"][:]))
    except Exception as e:
        print(f"  ERROR reading {dream3d_path}: {e}")
    return None


def read_stats(stats_path: str) -> dict | None:
    """Read mu, sigma from a _stats.txt file."""
    values = []
    with open(stats_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
    if len(values) >= 2:
        return {"mu": values[0], "sigma": values[1]}
    return None


def predict_grains(mu: float, sigma: float, volume: float = VOLUME) -> float:
    """Raw formula: N = V / E[V_grain], E[V] = (pi/6)*exp(3*mu + 4.5*sigma^2)."""
    e_vol = (math.pi / 6.0) * math.exp(3.0 * mu + 4.5 * sigma**2)
    return volume / e_vol


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    results = []

    print(f"{'Material':<25} {'ID':>5} {'Actual':>7} {'Predicted':>9} {'Ratio':>7} {'mu':>7} {'sigma':>7}")
    print("-" * 75)

    for material in MATERIALS:
        for sid in SAMPLE_IDS:
            dream3d_path = os.path.join(RVE_ROOT, material, str(sid), f"{material}_{sid}.dream3d")
            stats_path = os.path.join(STATS_ROOT, material, f"{material}_{sid}_stats.txt")

            if not os.path.exists(dream3d_path) or not os.path.exists(stats_path):
                continue

            stats = read_stats(stats_path)
            if stats is None:
                continue

            actual = get_grain_count(dream3d_path)
            if actual is None:
                continue

            predicted = predict_grains(stats["mu"], stats["sigma"])
            ratio = actual / predicted if predicted > 0 else float("nan")

            results.append({
                "material": material,
                "id": sid,
                "actual": actual,
                "predicted": predicted,
                "ratio": ratio,
                "mu": stats["mu"],
                "sigma": stats["sigma"],
            })

            print(
                f"{material:<25} {sid:>5} {actual:>7} {predicted:>9.1f} {ratio:>7.3f} "
                f"{stats['mu']:>7.4f} {stats['sigma']:>7.4f}"
            )

    if not results:
        print("\nNo results found. Check paths.")
        return

    # ─── Summary ─────────────────────────────────────────────────────────────
    ratios = [r["ratio"] for r in results]
    k = float(np.mean(ratios))
    k_std = float(np.std(ratios))
    k_median = float(np.median(ratios))

    print(f"\n{'='*75}")
    print(f"Samples analyzed: {len(results)}")
    print(f"Correction factor k (mean ± std): {k:.4f} ± {k_std:.4f}")
    print(f"Correction factor k (median):     {k_median:.4f}")
    print(f"\nCalibrated formula:")
    print(f"  n_grains ≈ {k:.3f} * {VOLUME:.0f} / ((π/6) * exp(3*mu + 4.5*sigma²))")
    print(f"\nFor target range [250, 1000] grains:")

    # Solve for bounds on 3*mu + 4.5*sigma^2
    log_term_min = math.log(k * VOLUME / (1000 * math.pi / 6))
    log_term_max = math.log(k * VOLUME / (250 * math.pi / 6))
    print(f"  Constraint: {log_term_min:.4f} <= 3*mu + 4.5*sigma² <= {log_term_max:.4f}")

    # Show distribution of actual grain counts
    actuals = [r["actual"] for r in results]
    print(f"\nActual grain count stats:")
    print(f"  min={min(actuals)}, max={max(actuals)}, mean={np.mean(actuals):.0f}, median={np.median(actuals):.0f}")
    in_range = sum(1 for a in actuals if 250 <= a <= 1000)
    print(f"  In [250, 1000]: {in_range}/{len(actuals)} ({100*in_range/len(actuals):.1f}%)")


if __name__ == "__main__":
    main()
