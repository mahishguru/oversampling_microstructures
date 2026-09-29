#!/usr/bin/env python3
"""Batch-extract 5 Gaussian-copula parameters from DREAM.3D mask statistics CSVs.

What this does (per your spec)
- Walk inside each folder under:
    data/measured
  and, for each material/temp_strain, look specifically in:
    <material>/<temp_strain>/OM/temp/temp_binary/

- Find all CSV files similar to grain_size_stats_batch.py (mask statistics).

- For all grains aggregated across those CSVs:
  1) Compute ESD (µm) from Area (px^2) and the um_per_px scale parsed from filename
  2) Compute Aspect Ratio = B/A (B=min(width,height), A=max(width,height))
  3) Fit marginals:
     - ln(ESD) ~ Normal(mu_log_esd, sigma_log_esd)
     - AR ~ Beta(alpha_ar, beta_ar) with bounds forced to [0,1]
  4) Map to latent normals and compute Gaussian copula correlation rho

- Save these 5 parameters into:
    STATS_COPULA_DIR = data/stats_copula

Output format (one file per temp_strain)
- <material>_<temp_strain>_copula_params.txt
  Lines: mu_log_esd, sigma_log_esd, alpha_ar, beta_ar, rho

Notes
- Uses SciPy because Beta fitting + CDF/PPF are easiest/standard there.
  Your environment currently prints a SciPy/NumPy version warning; if SciPy
  still imports successfully, this script will run.
"""

from __future__ import annotations

import argparse
import os
import re
import traceback

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm


ROOT_DIR = "data/measured"
STATS_COPULA_DIR = "data/stats_copula"


def _parse_um_per_px_from_filename(filename: str) -> float:
    """Parse um_per_px from filenames like:

    - 350_0.5_0.054_cropped_mask_statistics.csv
    - 450_2_0.107_extra1_cropped_mask_statistics.csv

    Strategy
    - Prefer 3rd underscore token (index 2) because that's how your database is named.
    - Fallback: pick the smallest positive float < 1.0 found in underscore tokens.
    """

    parts = os.path.basename(filename).split("_")
    if len(parts) >= 3:
        try:
            v = float(parts[2])
            if v > 0:
                return v
        except Exception:
            pass

    floats: list[float] = []
    for token in parts:
        token = re.sub(r"[^0-9.eE+-]", "", token)
        try:
            floats.append(float(token))
        except Exception:
            continue
    candidates = [v for v in floats if v > 0 and v < 1.0]
    if not candidates:
        raise ValueError(f"Could not parse um_per_px from filename: {filename}")
    return float(min(candidates))


def _collect_csvs(temp_binary_dir: str) -> list[str]:
    csvs: list[str] = []
    if not os.path.isdir(temp_binary_dir):
        return csvs
    for dirpath, _, filenames in os.walk(temp_binary_dir):
        for name in filenames:
            if name.endswith("_mask_statistics.csv") or name.endswith("_cropped_mask_statistics.csv"):
                csvs.append(os.path.join(dirpath, name))
    return sorted(csvs)


def _compute_esd_and_ar(df: pd.DataFrame, um_per_px: float) -> tuple[np.ndarray, np.ndarray]:
    if not {"Area", "Width", "Height"}.issubset(df.columns):
        raise KeyError("CSV must contain Area, Width, Height columns")

    areas = df["Area"].to_numpy(dtype=float)
    width_um = df["Width"].to_numpy(dtype=float) * um_per_px
    height_um = df["Height"].to_numpy(dtype=float) * um_per_px

    esd_um = 2.0 * np.sqrt(areas / np.pi) * um_per_px
    a = np.maximum(width_um, height_um)
    b = np.minimum(width_um, height_um)

    with np.errstate(divide="ignore", invalid="ignore"):
        ar = b / a

    # Clean like your example: ESD>0 and 0<AR<1.
    mask = np.isfinite(esd_um) & np.isfinite(ar) & (esd_um > 0) & (ar > 0) & (ar < 1)
    return esd_um[mask].astype(np.float64), ar[mask].astype(np.float64)


def _fit_copula_params(esd_um: np.ndarray, ar: np.ndarray) -> tuple[float, float, float, float, float]:
    if esd_um.size < 10:
        raise ValueError(f"Not enough grains to fit (n={esd_um.size})")

    log_esd = np.log(esd_um)
    mu_log_esd, sigma_log_esd = norm.fit(log_esd)

    # Force beta support to [0,1]
    alpha_ar, beta_ar, _, _ = beta_dist.fit(ar, floc=0, fscale=1)

    # Map to uniforms
    u_esd = norm.cdf(log_esd, loc=mu_log_esd, scale=sigma_log_esd)
    u_ar = beta_dist.cdf(ar, a=alpha_ar, b=beta_ar, loc=0, scale=1)

    # Map to latent normals
    eps = 1e-6
    u_esd = np.clip(u_esd, eps, 1.0 - eps)
    u_ar = np.clip(u_ar, eps, 1.0 - eps)

    z_esd = norm.ppf(u_esd)
    z_ar = norm.ppf(u_ar)

    # Gaussian copula coupling parameter = Pearson correlation in latent normal space
    rho = float(np.corrcoef(z_esd, z_ar)[0, 1])

    return float(mu_log_esd), float(sigma_log_esd), float(alpha_ar), float(beta_ar), rho


def _find_temp_strain_dirs(root: str) -> list[str]:
    material_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    temp_strain_dirs: list[str] = []
    for mat_dir in material_dirs:
        for d in os.listdir(mat_dir):
            ts_dir = os.path.join(mat_dir, d)
            if os.path.isdir(ts_dir):
                temp_strain_dirs.append(ts_dir)
    return sorted(temp_strain_dirs)


def _write_params(out_dir: str, material: str, temp_strain: str, params: tuple[float, float, float, float, float]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{material}_{temp_strain}_copula_params.txt")
    mu_log_esd, sigma_log_esd, alpha_ar, beta_ar, rho = params
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# mu_log_esd sigma_log_esd alpha_ar beta_ar rho\n")
        f.write(f"{mu_log_esd}\n{sigma_log_esd}\n{alpha_ar}\n{beta_ar}\n{rho}\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract copula parameters from database temp_binary CSVs")
    ap.add_argument("--root", type=str, default=ROOT_DIR)
    ap.add_argument("--out", type=str, default=STATS_COPULA_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N temp_strain folders (0=all)")
    args = ap.parse_args()

    ts_dirs = _find_temp_strain_dirs(args.root)
    if args.limit and args.limit > 0:
        ts_dirs = ts_dirs[: args.limit]

    ok = 0
    skipped = 0
    failed = 0

    for ts_dir in ts_dirs:
        rel = os.path.relpath(ts_dir, args.root)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            skipped += 1
            continue
        material, temp_strain = parts[0], parts[1]

        temp_binary_dir = os.path.join(ts_dir, "OM", "temp", "temp_binary")
        csvs = _collect_csvs(temp_binary_dir)
        if not csvs:
            skipped += 1
            continue

        esd_all: list[np.ndarray] = []
        ar_all: list[np.ndarray] = []

        try:
            for csv_path in csvs:
                um_per_px = _parse_um_per_px_from_filename(csv_path)
                df = pd.read_csv(csv_path)
                esd_um, ar = _compute_esd_and_ar(df, um_per_px)
                if esd_um.size:
                    esd_all.append(esd_um)
                    ar_all.append(ar)

            if not esd_all:
                print(f"No valid grains after cleaning: {material}/{temp_strain}")
                skipped += 1
                continue

            esd = np.concatenate(esd_all)
            ar = np.concatenate(ar_all)

            params = _fit_copula_params(esd, ar)
            out_path = _write_params(args.out, material, temp_strain, params)

            mu_log_esd, sigma_log_esd, alpha_ar, beta_ar, rho = params
            print(
                f"OK {material}/{temp_strain}: n={esd.size} -> mu={mu_log_esd:.6f}, sigma={sigma_log_esd:.6f}, "
                f"alpha={alpha_ar:.6f}, beta={beta_ar:.6f}, rho={rho:.6f}\n  Wrote: {out_path}"
            )
            ok += 1

        except Exception as e:
            failed += 1
            print(
                f"FAIL {material}/{temp_strain}: {e}\n"
                f"  temp_binary_dir={temp_binary_dir}\n"
                f"  {traceback.format_exc()}"
            )

    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
