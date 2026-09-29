#!/usr/bin/env python3
"""Generate DREAM.3D-style stats tables from *sampled copula parameter* files.

Input layout (per your description)
- IN_ROOT/material/*.txt
  Example:
    data/stats_copula_sampled/AZ31_extruded/AZ31_extruded_1000.txt

Each input .txt contains 5 copula parameters (one per line), with an optional header:
  mu_log_esd, sigma_log_esd, alpha_ar, beta_ar, rho

For each input file, this script:
1) Samples N synthetic grains using a Gaussian copula:
   - latent (z1,z2) ~ N(0, cov=[[1,rho],[rho,1]])
   - u = Phi(z)
   - ln(ESD) = NormalPPF(u1; mu_log_esd, sigma_log_esd) -> ESD = exp(ln(ESD))
   - AR = BetaPPF(u2; alpha_ar, beta_ar)

2) Produces the same "report" outputs as grain_size_stats_batch.py:
   - global mu_lnESD, sigma_lnESD fit from synthetic data
   - global aspect_mu, aspect_sigma
   - size bins defined by +/- sigma cutoffs in log space
   - per-bin Beta(alpha,beta) fit (SciPy MLE) on aspect ratio values

3) Writes outputs into OUT_ROOT/material/ with matching base names:
   - <base>_stats.txt
   - <base>_stats_aspect_beta.txt

Output layout
- data/stats_combined_sampled/<material>/

Notes
- Uses SciPy (norm/beta CDF/PPF + beta.fit). Your environment currently warns
  about SciPy vs NumPy versions, but SciPy imports and works.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import tempfile
import traceback
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import gaussian_kde
from scipy.stats import norm


DEFAULT_IN_ROOT = "data/stats_copula_sampled"
DEFAULT_OUT_ROOT = "data/stats_combined_sampled"


@dataclass(frozen=True)
class CopulaParams:
    mu_log_esd: float
    sigma_log_esd: float
    alpha_ar: float
    beta_ar: float
    rho: float


def _list_material_dirs(in_root: str) -> list[str]:
    if not os.path.isdir(in_root):
        raise FileNotFoundError(f"Input root not found: {in_root}")
    return sorted(
        os.path.join(in_root, d)
        for d in os.listdir(in_root)
        if os.path.isdir(os.path.join(in_root, d))
    )


def _read_copula_params(txt_path: str) -> CopulaParams:
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()

    vals: list[float] = []
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # tolerate "key: value" too
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        # remove any trailing comments
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        try:
            vals.append(float(line))
        except Exception:
            continue

    if len(vals) < 5:
        raise ValueError(f"Expected 5 numeric values in {txt_path}, got {len(vals)}")

    mu, sigma, a, b, rho = vals[:5]
    if sigma <= 0:
        raise ValueError(f"sigma_log_esd must be > 0 in {txt_path}")
    if a <= 0 or b <= 0:
        raise ValueError(f"alpha_ar and beta_ar must be > 0 in {txt_path}")
    if rho < -1 or rho > 1:
        raise ValueError(f"rho must be in [-1,1] in {txt_path}")

    return CopulaParams(mu, sigma, a, b, rho)


def _sample_synthetic_grains(
    *,
    params: CopulaParams,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # Sample correlated latent normals.
    rho = params.rho
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)

    z_esd = z[:, 0]
    z_ar = z[:, 1]

    # Uniform percentiles
    u_esd = norm.cdf(z_esd)
    u_ar = norm.cdf(z_ar)

    # Avoid exact 0/1.
    eps = 1e-10
    u_esd = np.clip(u_esd, eps, 1.0 - eps)
    u_ar = np.clip(u_ar, eps, 1.0 - eps)

    # Map to physical space
    log_esd = norm.ppf(u_esd, loc=params.mu_log_esd, scale=params.sigma_log_esd)
    esd = np.exp(log_esd)

    ar = beta_dist.ppf(u_ar, a=params.alpha_ar, b=params.beta_ar, loc=0, scale=1)

    # Clean
    mask = np.isfinite(esd) & np.isfinite(ar) & (esd > 0) & (ar > 0) & (ar < 1)
    return esd[mask].astype(np.float64), ar[mask].astype(np.float64)


def _bin_edges_from_mu_sigma(
    mu_lnESD: float,
    sigma_lnESD: float,
    sigma_cutoff_min: float,
    sigma_cutoff_max: float,
    bin_step: float,
) -> np.ndarray:
    min_esd = float(np.exp(mu_lnESD - sigma_cutoff_min * sigma_lnESD))
    max_esd = float(np.exp(mu_lnESD + sigma_cutoff_max * sigma_lnESD))
    num_bins = int(np.floor((max_esd - min_esd) / bin_step)) + 1
    return np.linspace(min_esd, max_esd, num_bins + 1)


def _beta_fit_per_bin(
    *,
    esd_um: np.ndarray,
    aspect_ratio: np.ndarray,
    bin_edges: np.ndarray,
    min_count: int,
    fill_zeros_with_kde: bool,
) -> pd.DataFrame:
    df = pd.DataFrame({"esd_um": esd_um, "aspect_ratio": aspect_ratio})
    df = df[(df["esd_um"] > 0) & (df["aspect_ratio"] > 0) & (df["aspect_ratio"] < 1)].copy()

    df["diameter_bin"] = pd.cut(df["esd_um"], bins=bin_edges, right=True, include_lowest=False)
    df = df.dropna(subset=["diameter_bin", "aspect_ratio"])

    def beta_fit_stats(subdf: pd.DataFrame) -> pd.Series:
        ar = subdf["aspect_ratio"].dropna()
        n = int(len(ar))
        if n < min_count:
            return pd.Series({"grain_count": n, "alpha": np.nan, "beta": np.nan})
        ar = ar[(ar > 0) & (ar < 1)]
        if len(ar) < 2:
            return pd.Series({"grain_count": n, "alpha": np.nan, "beta": np.nan})
        try:
            a, b, _, _ = beta_dist.fit(ar, floc=0, fscale=1)
        except Exception:
            return pd.Series({"grain_count": n, "alpha": np.nan, "beta": np.nan})
        return pd.Series({"grain_count": n, "alpha": float(a), "beta": float(b)})

    binned = (
        df.groupby("diameter_bin", observed=True, group_keys=False)
        .apply(beta_fit_stats)
        .reset_index()
    )

    # Ensure every bin is present (even if empty).
    expected_bins = df["diameter_bin"].cat.categories
    binned = (
        binned.set_index("diameter_bin")
        .reindex(expected_bins)
        .rename_axis("diameter_bin")
        .reset_index()
    )

    binned["grain_count"] = binned["grain_count"].fillna(0).astype(int)
    binned["alpha"] = binned["alpha"].fillna(0.0)
    binned["beta"] = binned["beta"].fillna(0.0)

    if fill_zeros_with_kde:
        nonzero_pairs = binned[(binned["alpha"] > 0) & (binned["beta"] > 0)][["alpha", "beta"]].to_numpy()
        unique_pairs = np.unique(nonzero_pairs, axis=0) if len(nonzero_pairs) else np.empty((0, 2))
        used_kde = False
        kde = None
        if len(nonzero_pairs) > 1 and len(unique_pairs) > 2:
            try:
                kde = gaussian_kde(nonzero_pairs.T)
                used_kde = True
            except Exception:
                used_kde = False

        for idx, row in binned.iterrows():
            if row["alpha"] == 0.0 or row["beta"] == 0.0:
                if used_kde and kde is not None:
                    for _ in range(100):
                        sampled = kde.resample(1).flatten()
                        if sampled[0] > 0 and sampled[1] > 0:
                            binned.at[idx, "alpha"] = float(sampled[0])
                            binned.at[idx, "beta"] = float(sampled[1])
                            break
                elif len(nonzero_pairs) > 0:
                    sampled_pair = nonzero_pairs[np.random.choice(len(nonzero_pairs))]
                    binned.at[idx, "alpha"] = float(sampled_pair[0])
                    binned.at[idx, "beta"] = float(sampled_pair[1])

    return binned


def _write_outputs(
    *,
    out_dir: str,
    base: str,
    mu_lnESD: float,
    sigma_lnESD: float,
    aspect_mu: float,
    aspect_sigma: float,
    num_bins: int,
    binned: pd.DataFrame,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(out_dir, f"{base}_stats.txt")
    beta_path = os.path.join(out_dir, f"{base}_stats_aspect_beta.txt")

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("# mu_lnESD sigma_lnESD aspect_mu aspect_sigma num_bins\n")
        f.write(f"{mu_lnESD}\n{sigma_lnESD}\n{aspect_mu}\n{aspect_sigma}\n{num_bins}\n")

    np.savetxt(beta_path, binned[["alpha", "beta"]].to_numpy(), fmt="%.6f")

    return stats_path, beta_path


def _stable_seed(base_seed: int, key: str) -> int:
    """Deterministic per-file seed derived from a run seed + file path."""

    return (int(base_seed) + (zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)) % (2**32)


def _process_one(
    *,
    txt_path: str,
    material: str,
    out_mat_dir: str,
    n: int,
    base_seed: int,
    sigma_cutoff_min: float,
    sigma_cutoff_max: float,
    bin_step: float,
    min_count: int,
    fill_zeros_with_kde: bool,
    tmp_root: str,
    keep_temp_csv: bool,
) -> tuple[bool, str]:
    """Worker: process exactly one sampled-copula file."""

    base = os.path.splitext(os.path.basename(txt_path))[0]
    try:
        params = _read_copula_params(txt_path)

        seed = _stable_seed(base_seed, f"{material}/{base}")
        rng = np.random.default_rng(seed)

        esd_um, ar = _sample_synthetic_grains(params=params, n=int(n), rng=rng)
        if esd_um.size < 10:
            raise ValueError(f"Too few valid synthetic grains after cleaning: n={esd_um.size}")

        # Store synthetic grains in a temp file (fast binary). This satisfies the
        # "store the data in a temp file" requirement without heavy CSV IO.
        tmp_npz = os.path.join(tmp_root, f"{material}_{base}_synthetic_100k.npz")
        np.savez(tmp_npz, ESD_um=esd_um, Aspect_Ratio=ar)

        # Optional: keep a human-readable CSV next to outputs.
        if keep_temp_csv:
            os.makedirs(out_mat_dir, exist_ok=True)
            kept_csv = os.path.join(out_mat_dir, f"{base}_synthetic_100k.csv")
            pd.DataFrame({"ESD_um": esd_um, "Aspect_Ratio": ar}).to_csv(kept_csv, index=False)

        # Fit mu/sigma from synthetic, to match grain_size_stats_batch.py behavior.
        log_esd = np.log(esd_um)
        mu_lnESD = float(np.mean(log_esd))
        sigma_lnESD = float(np.std(log_esd))

        aspect_mu = float(np.mean(ar))
        aspect_sigma = float(np.std(ar))

        bin_edges = _bin_edges_from_mu_sigma(
            mu_lnESD,
            sigma_lnESD,
            float(sigma_cutoff_min),
            float(sigma_cutoff_max),
            float(bin_step),
        )
        num_bins = int(len(bin_edges) - 1)

        binned = _beta_fit_per_bin(
            esd_um=esd_um,
            aspect_ratio=ar,
            bin_edges=bin_edges,
            min_count=int(min_count),
            fill_zeros_with_kde=bool(fill_zeros_with_kde),
        )

        stats_path, beta_path = _write_outputs(
            out_dir=out_mat_dir,
            base=base,
            mu_lnESD=mu_lnESD,
            sigma_lnESD=sigma_lnESD,
            aspect_mu=aspect_mu,
            aspect_sigma=aspect_sigma,
            num_bins=num_bins,
            binned=binned,
        )

        msg = (
            f"OK {material}/{base}: n={esd_um.size} bins={num_bins} ->\n"
            f"  Wrote: {stats_path}\n"
            f"  Wrote: {beta_path}"
        )
        return True, msg

    except Exception as e:
        msg = (
            f"FAIL {material}/{base}: {e}\n"
            f"  src={txt_path}\n"
            f"  {traceback.format_exc()}"
        )
        return False, msg


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample synthetic grains from sampled copulas and write DREAM.3D-style stats tables")
    ap.add_argument("--in-root", type=str, default=DEFAULT_IN_ROOT)
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-cutoff-min", type=float, default=3.0)
    ap.add_argument("--sigma-cutoff-max", type=float, default=3.0)
    ap.add_argument("--bin-step", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--fill-zeros-with-kde", action="store_true", default=True)
    ap.add_argument("--no-kde-fill", dest="fill_zeros_with_kde", action="store_false")
    ap.add_argument("--limit-materials", type=int, default=0)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0=auto=os.cpu_count()).",
    )
    ap.add_argument(
        "--keep-temp-csv",
        action="store_true",
        help="If set, keep the per-input synthetic grains CSV next to outputs (otherwise uses a temporary file that is deleted).",
    )
    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    mat_dirs = _list_material_dirs(args.in_root)
    if args.limit_materials and args.limit_materials > 0:
        mat_dirs = mat_dirs[: args.limit_materials]

    ok = 0
    failed = 0
    skipped = 0

    # Build task list first (so parallel workers are fed consistently).
    tasks: list[tuple[str, str, str]] = []  # (material, out_mat_dir, txt_path)
    for mat_dir in mat_dirs:
        material = os.path.basename(mat_dir)
        out_mat_dir = os.path.join(args.out_root, material)

        txt_files = sorted(
            os.path.join(mat_dir, f)
            for f in os.listdir(mat_dir)
            if f.lower().endswith(".txt")
        )
        if args.limit_files and args.limit_files > 0:
            txt_files = txt_files[: args.limit_files]
        if not txt_files:
            skipped += 1
            continue
        for txt_path in txt_files:
            tasks.append((material, out_mat_dir, txt_path))

    if not tasks:
        print("Done. No input files found.")
        return

    # One temp dir for the whole run.
    with tempfile.TemporaryDirectory(prefix="synthetic_grains_") as tmp_root:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _process_one,
                    txt_path=txt_path,
                    material=material,
                    out_mat_dir=out_mat_dir,
                    n=int(args.n),
                    base_seed=int(args.seed),
                    sigma_cutoff_min=float(args.sigma_cutoff_min),
                    sigma_cutoff_max=float(args.sigma_cutoff_max),
                    bin_step=float(args.bin_step),
                    min_count=int(args.min_count),
                    fill_zeros_with_kde=bool(args.fill_zeros_with_kde),
                    tmp_root=tmp_root,
                    keep_temp_csv=bool(args.keep_temp_csv),
                )
                for (material, out_mat_dir, txt_path) in tasks
            ]

            for fut in concurrent.futures.as_completed(futures):
                ok_one, msg = fut.result()
                print(msg)
                if ok_one:
                    ok += 1
                else:
                    failed += 1

    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
