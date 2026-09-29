#!/usr/bin/env python3
"""Filter stats_combined_sampled files by predicted grain count.

Reads each *_stats.txt file, predicts the number of grains DREAM3D would
generate based on (mu, sigma) and the RVE volume, and removes files whose
predicted grain count falls outside the target range [min_grains, max_grains].

The prediction formula (empirically validated, median error <5%):
    n_grains ≈ V_RVE / ((π/6) * exp(3*mu + 4.5*sigma²))

Usage:
    python 03b_filter_by_grain_count.py --root /path/to/stats_combined_sampled
    python 03b_filter_by_grain_count.py --dry-run   # preview without deleting
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys


# ─── RVE geometry (must match template.json used in 04_generate_rves.py) ─────
NX, NY, NZ = 300, 300, 1
DX, DY, DZ = 2.0, 2.0, 2.0  # µm per voxel
VOLUME = NX * DX * NY * DY * NZ * DZ  # 720,000 µm³

DEFAULT_ROOT = "data/stats_combined_sampled"


def predict_grain_count(mu: float, sigma: float, volume: float = VOLUME) -> float:
    """Predict number of grains in the RVE.

    N ≈ V / E[V_grain]
    where E[V_grain] = (π/6) * exp(3µ + 4.5σ²) for a log-normal ESD distribution.
    """
    e_vol = (math.pi / 6.0) * math.exp(3.0 * mu + 4.5 * sigma**2)
    if e_vol <= 0:
        return 0.0
    return volume / e_vol


def read_mu_sigma(stats_path: str) -> tuple[float, float] | None:
    """Read mu and sigma from a _stats.txt file (first two numeric lines)."""
    values: list[float] = []
    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
            if len(values) == 2:
                break
    if len(values) >= 2 and values[1] > 0:
        return values[0], values[1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Filter stats_combined_sampled by predicted grain count."
    )
    ap.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Root directory containing material subfolders with *_stats.txt files.",
    )
    ap.add_argument(
        "--min-grains",
        type=int,
        default=250,
        help="Minimum acceptable grain count (default: 250).",
    )
    ap.add_argument(
        "--max-grains",
        type=int,
        default=1000,
        help="Maximum acceptable grain count (default: 1000).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without actually deleting.",
    )
    ap.add_argument(
        "--move-to",
        type=str,
        default=None,
        help="Instead of deleting, move rejected files to this directory (preserving material subfolder structure).",
    )
    ap.add_argument(
        "--also-filter-copula",
        type=str,
        default=None,
        help="Also remove/move corresponding files from stats_copula_sampled dir.",
    )
    ap.add_argument(
        "--also-filter-odf",
        type=str,
        default=None,
        help="Also remove/move corresponding ODF files from odf_harmonics_sampled dir.",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: root directory not found: {args.root}")
        return 1

    min_n = args.min_grains
    max_n = args.max_grains

    # Collect all material subdirs
    mat_dirs = sorted(
        os.path.join(args.root, d)
        for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    )

    total = 0
    kept = 0
    removed = 0
    errors = 0

    for mat_dir in mat_dirs:
        material = os.path.basename(mat_dir)
        stats_files = sorted(
            f for f in os.listdir(mat_dir) if f.endswith("_stats.txt") and not f.endswith("_aspect_beta.txt")
        )

        for stats_file in stats_files:
            stats_path = os.path.join(mat_dir, stats_file)
            total += 1

            result = read_mu_sigma(stats_path)
            if result is None:
                errors += 1
                print(f"  WARN: could not read mu/sigma from {stats_path}")
                continue

            mu, sigma = result
            n_pred = predict_grain_count(mu, sigma)

            if min_n <= n_pred <= max_n:
                kept += 1
                continue

            # This file is outside the range → remove/move
            removed += 1

            # Derive the base name and companion aspect_beta file
            # e.g. AZ31_extruded_1000_stats.txt → AZ31_extruded_1000
            base = stats_file.replace("_stats.txt", "")
            aspect_file = f"{base}_stats_aspect_beta.txt"
            aspect_path = os.path.join(mat_dir, aspect_file)

            if args.dry_run:
                print(f"  REJECT {material}/{base}: n_pred={n_pred:.0f} (mu={mu:.4f}, sigma={sigma:.4f})")
                continue

            # Remove or move
            files_to_remove = [stats_path]
            if os.path.exists(aspect_path):
                files_to_remove.append(aspect_path)

            if args.move_to:
                dest_dir = os.path.join(args.move_to, material)
                os.makedirs(dest_dir, exist_ok=True)
                for fp in files_to_remove:
                    shutil.move(fp, os.path.join(dest_dir, os.path.basename(fp)))
            else:
                for fp in files_to_remove:
                    os.remove(fp)

            # Also handle copula sampled file
            if args.also_filter_copula:
                copula_path = os.path.join(args.also_filter_copula, material, f"{base}.txt")
                if os.path.exists(copula_path):
                    if args.move_to:
                        dest_dir = os.path.join(args.move_to, "copula_sampled", material)
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.move(copula_path, os.path.join(dest_dir, os.path.basename(copula_path)))
                    else:
                        os.remove(copula_path)

            # Also handle ODF sampled files
            if args.also_filter_odf:
                odf_dir = os.path.join(args.also_filter_odf, material)
                if os.path.isdir(odf_dir):
                    # ODF files: base.txt, base_ODF.txt, base_pf.jpg
                    odf_files = [
                        os.path.join(odf_dir, f"{base}.txt"),
                        os.path.join(odf_dir, f"{base}_ODF.txt"),
                        os.path.join(odf_dir, f"{base}_pf.jpg"),
                    ]
                    for odf_f in odf_files:
                        if os.path.exists(odf_f):
                            if args.move_to:
                                dest_dir = os.path.join(args.move_to, "odf_sampled", material)
                                os.makedirs(dest_dir, exist_ok=True)
                                shutil.move(odf_f, os.path.join(dest_dir, os.path.basename(odf_f)))
                            else:
                                os.remove(odf_f)

    # Summary
    print(f"\n{'='*60}")
    print(f"Filter: [{min_n}, {max_n}] grains | V_RVE = {VOLUME:.0f} µm³")
    print(f"Total samples scanned:  {total}")
    print(f"Kept (in range):        {kept} ({100*kept/max(total,1):.1f}%)")
    print(f"Removed (out of range): {removed} ({100*removed/max(total,1):.1f}%)")
    if errors:
        print(f"Errors (skipped):       {errors}")
    if args.dry_run:
        print("(DRY RUN — no files were modified)")
    elif args.move_to:
        print(f"Rejected files moved to: {args.move_to}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
