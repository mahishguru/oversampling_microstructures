import os
import glob
import argparse
import math
import numpy as np
from scipy.stats import gaussian_kde

# ─── Grain count filter (must match RVE geometry in template.json) ────────────
_NX, _NY, _NZ = 300, 300, 1
_DX, _DY, _DZ = 2.0, 2.0, 2.0  # µm per voxel
_RVE_VOLUME = _NX * _DX * _NY * _DY * _NZ * _DZ  # 720,000 µm³
_MIN_GRAINS = 250
_MAX_GRAINS = 1000


def read_stats_file(filename: str):
    """Read a stats_copula file.

    Expected header (typical):
        # mu_log_esd sigma_log_esd alpha_ar beta_ar rho

    Returns:
        (header: str, values: list[float] of length 5) or None
    """

    header = None
    floats: list[float] = []

    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if header is None and stripped.startswith("#"):
                header = stripped
                continue

            # Most files are one-number-per-line, but token-scan makes it robust.
            for part in stripped.split():
                try:
                    floats.append(float(part))
                except ValueError:
                    continue

    if len(floats) < 5:
        return None

    values = [float(x) for x in floats[:5]]

    if header is None:
        header = "# mu_log_esd sigma_log_esd alpha_ar beta_ar rho"
    else:
        # Normalize to the 5-field header if extra tokens exist.
        if "rho" in header:
            header = "# mu_log_esd sigma_log_esd alpha_ar beta_ar rho"

    return header, values


def get_group_key(filename: str) -> str:
    base = os.path.basename(filename)
    if "extruded_heattreated" in base:
        idx = base.find("extruded_heattreated")
        key = base[: idx + len("extruded_heattreated")]
    elif "extruded" in base:
        idx = base.find("extruded")
        key = base[: idx + len("extruded")]
    else:
        key = base.split(".")[0]
    return key.rstrip("_")


def count_target_samples_for_group(odf_harmonics_sampled_dir: str, group_key: str) -> int:
    group_odf_dir = os.path.join(odf_harmonics_sampled_dir, group_key)
    if not os.path.isdir(group_odf_dir):
        return 0

    odf_txt_files = [
        p for p in glob.glob(os.path.join(group_odf_dir, "*.txt")) if not p.endswith("ODF.txt")
    ]
    return len(odf_txt_files)


def resample_with_constraints(kde: gaussian_kde, n_samples: int) -> np.ndarray | None:
    """Resample from KDE until we collect n_samples satisfying basic constraints.

    Constraints:
      - sigma_log_esd > 0
      - alpha_ar > 0
      - beta_ar > 0
      - rho in [-1, 1]
      - all finite
      - predicted grain count in [_MIN_GRAINS, _MAX_GRAINS]

    Returns an (n_samples, 5) array, or None if unable to collect enough samples.
    """

    if n_samples <= 0:
        return np.empty((0, 5), dtype=float)

    collected: list[np.ndarray] = []
    remaining = n_samples

    max_rounds = 200
    oversample_factor = 5.0

    for _ in range(max_rounds):
        batch_n = int(np.ceil(max(remaining, 1) * oversample_factor))
        batch = kde.resample(batch_n).T  # (batch_n, 5)

        finite_mask = np.isfinite(batch).all(axis=1)
        sigma_pos = batch[:, 1] > 0
        alpha_pos = batch[:, 2] > 0
        beta_pos = batch[:, 3] > 0
        rho_ok = (batch[:, 4] >= -1.0) & (batch[:, 4] <= 1.0)

        # Grain count constraint
        mu_arr = batch[:, 0]
        sig_arr = batch[:, 1]
        e_vol = (np.pi / 6.0) * np.exp(3.0 * mu_arr + 4.5 * sig_arr**2)
        n_grains = _RVE_VOLUME / np.where(e_vol > 0, e_vol, 1e30)
        grain_ok = (n_grains >= _MIN_GRAINS) & (n_grains <= _MAX_GRAINS)

        valid = batch[finite_mask & sigma_pos & alpha_pos & beta_pos & rho_ok & grain_ok]

        if valid.size:
            take = min(remaining, valid.shape[0])
            collected.append(valid[:take])
            remaining -= take
            if remaining == 0:
                break

    if remaining != 0:
        return None

    return np.vstack(collected)


def _nearest_pd_cov(cov: np.ndarray) -> np.ndarray:
    """Return a positive-definite covariance via diagonal loading.

    For small sample sizes (e.g., 4 points in 5D), the empirical covariance is
    rank-deficient. We fix this by adding a data-scaled jitter to the diagonal.
    """

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)

    d = cov.shape[0]
    if d == 0:
        return cov

    tr = float(np.trace(cov))
    scale = tr / d if np.isfinite(tr) and tr > 0 else 1.0
    min_eig_floor = 1e-6 * scale

    try:
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = float(np.min(eigvals))
    except np.linalg.LinAlgError:
        min_eig = -1.0

    if not np.isfinite(min_eig):
        min_eig = -1.0

    if min_eig < min_eig_floor:
        cov = cov + np.eye(d) * (min_eig_floor - min_eig)

    # If still not PD due to numerical issues, increase jitter geometrically.
    jitter = min_eig_floor
    for _ in range(8):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            cov = cov + np.eye(d) * jitter
            jitter *= 10.0

    return cov


def resample_gaussian_with_constraints(original_data: np.ndarray, n_samples: int) -> np.ndarray | None:
    """Fallback sampler using a regularized multivariate normal.

    Uses empirical mean + (regularized) empirical covariance, then rejection
    sampling to enforce constraints.
    """

    if n_samples <= 0:
        return np.empty((0, 5), dtype=float)

    mean = np.mean(original_data, axis=0)
    cov = np.cov(original_data, rowvar=False)
    cov = _nearest_pd_cov(cov)

    collected: list[np.ndarray] = []
    remaining = n_samples

    max_rounds = 200
    oversample_factor = 5.0

    for _ in range(max_rounds):
        batch_n = int(np.ceil(max(remaining, 1) * oversample_factor))
        try:
            batch = np.random.multivariate_normal(mean, cov, size=batch_n)
        except ValueError:
            # As a last resort, add more diagonal loading.
            cov = _nearest_pd_cov(cov + np.eye(cov.shape[0]) * (1e-3 * float(np.trace(cov)) + 1e-6))
            try:
                batch = np.random.multivariate_normal(mean, cov, size=batch_n)
            except Exception:
                return None

        finite_mask = np.isfinite(batch).all(axis=1)
        sigma_pos = batch[:, 1] > 0
        alpha_pos = batch[:, 2] > 0
        beta_pos = batch[:, 3] > 0
        rho_ok = (batch[:, 4] >= -1.0) & (batch[:, 4] <= 1.0)

        # Grain count constraint
        mu_arr = batch[:, 0]
        sig_arr = batch[:, 1]
        e_vol = (np.pi / 6.0) * np.exp(3.0 * mu_arr + 4.5 * sig_arr**2)
        n_grains = _RVE_VOLUME / np.where(e_vol > 0, e_vol, 1e30)
        grain_ok = (n_grains >= _MIN_GRAINS) & (n_grains <= _MAX_GRAINS)

        valid = batch[finite_mask & sigma_pos & alpha_pos & beta_pos & rho_ok & grain_ok]

        if valid.size:
            take = min(remaining, valid.shape[0])
            collected.append(valid[:take])
            remaining -= take
            if remaining == 0:
                break

    if remaining != 0:
        return None

    return np.vstack(collected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch sample 5D copula stats (mu_log_esd, sigma_log_esd, alpha_ar, beta_ar, rho) "
            "using a per-group Gaussian KDE, writing one sampled file per sampled ODF harmonic."
        )
    )
    parser.add_argument(
        "--stats-dir",
        default="data/stats_copula",
        help="Input directory containing stats_copula *.txt files.",
    )
    parser.add_argument(
        "--sampled-dir",
        default="data/stats_copula_sampled",
        help="Output directory where sampled stats files are written (one subfolder per group).",
    )
    parser.add_argument(
        "--odf-harmonics-sampled-dir",
        default="data/odf_harmonics_sampled",
        help="Directory with per-group sampled ODF harmonics folders used to determine sample counts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print reasons when a group is skipped.",
    )
    parser.add_argument(
        "--small-group-fallback",
        choices=["skip", "gaussian"],
        default="gaussian",
        help=(
            "What to do when a group has too few unique points for a stable 5D KDE. "
            "'gaussian' uses a regularized multivariate normal fallback; 'skip' leaves it out."
        ),
    )
    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.stats_dir, "*.txt"))
    if not all_files:
        print(f"No .txt files found in stats_dir: {args.stats_dir}")
        return 1

    groups: dict[str, list[str]] = {}
    for path in all_files:
        key = get_group_key(path)
        groups.setdefault(key, []).append(path)

    os.makedirs(args.sampled_dir, exist_ok=True)

    for group_key, files in groups.items():
        data = []
        headers = []

        for filename in files:
            parsed = read_stats_file(filename)
            if parsed is None:
                continue
            header, values = parsed
            headers.append(header)
            data.append(values)

        if not data:
            if args.verbose:
                print(f"Skipping group '{group_key}': no readable 5-value stats files.")
            continue

        group_header = headers[0] if headers else "# mu_log_esd sigma_log_esd alpha_ar beta_ar rho"

        original_data = np.array(data, dtype=float)

        # For a 5D KDE, require enough distinct points to avoid singular covariance.
        min_points = 6
        if original_data.shape[0] < min_points or np.unique(original_data, axis=0).shape[0] < min_points:
            uniq = np.unique(original_data, axis=0).shape[0]
            if args.small_group_fallback == "skip":
                if args.verbose:
                    print(
                        f"Skipping group '{group_key}': need >= {min_points} unique points for 5D KDE; "
                        f"got n={original_data.shape[0]}, unique={uniq}."
                    )
                continue
            if args.verbose:
                print(
                    f"Group '{group_key}': using GAUSSIAN fallback (n={original_data.shape[0]}, unique={uniq} < {min_points})."
                )

            n_samples = count_target_samples_for_group(args.odf_harmonics_sampled_dir, group_key)
            if n_samples < 1:
                if args.verbose:
                    print(
                        f"Skipping group '{group_key}': missing/empty ODF sampled directory for sample count. "
                        f"Expected: {os.path.join(args.odf_harmonics_sampled_dir, group_key)}"
                    )
                continue

            synthetic_samples = resample_gaussian_with_constraints(original_data, n_samples)
            if synthetic_samples is None:
                print(
                    f"Skipping group '{group_key}': gaussian fallback could not draw {n_samples} valid samples "
                    "(constraints: sigma>0, alpha>0, beta>0, rho in [-1,1])."
                )
                continue

            out_dir = os.path.join(args.sampled_dir, group_key)
            os.makedirs(out_dir, exist_ok=True)

            for i, sample in enumerate(synthetic_samples, start=1):
                out_file = os.path.join(out_dir, f"{group_key}_{i}.txt")
                with open(out_file, "w") as f:
                    f.write(f"{group_header}\n")
                    f.write(f"{sample[0]}\n")
                    f.write(f"{sample[1]}\n")
                    f.write(f"{sample[2]}\n")
                    f.write(f"{sample[3]}\n")
                    f.write(f"{sample[4]}\n")

            print(f"Group '{group_key}': wrote {n_samples} samples to '{out_dir}' (gaussian fallback)")
            continue

        try:
            kde = gaussian_kde(original_data.T)
        except np.linalg.LinAlgError:
            print(f"Skipping group '{group_key}' due to singular covariance matrix (degenerate data).")
            continue

        n_samples = count_target_samples_for_group(args.odf_harmonics_sampled_dir, group_key)
        if n_samples < 1:
            if args.verbose:
                print(
                    f"Skipping group '{group_key}': missing/empty ODF sampled directory for sample count. "
                    f"Expected: {os.path.join(args.odf_harmonics_sampled_dir, group_key)}"
                )
            continue

        synthetic_samples = resample_with_constraints(kde, n_samples)
        if synthetic_samples is None:
            print(
                f"Skipping group '{group_key}': could not draw {n_samples} valid samples "
                "(constraints: sigma>0, alpha>0, beta>0, rho in [-1,1])."
            )
            continue

        out_dir = os.path.join(args.sampled_dir, group_key)
        os.makedirs(out_dir, exist_ok=True)

        for i, sample in enumerate(synthetic_samples, start=1):
            out_file = os.path.join(out_dir, f"{group_key}_{i}.txt")
            with open(out_file, "w") as f:
                f.write(f"{group_header}\n")
                f.write(f"{sample[0]}\n")
                f.write(f"{sample[1]}\n")
                f.write(f"{sample[2]}\n")
                f.write(f"{sample[3]}\n")
                f.write(f"{sample[4]}\n")

        print(f"Group '{group_key}': wrote {n_samples} samples to '{out_dir}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
