#!/usr/bin/env python3
"""Generate a synthetic stand-in for the measured GSH coefficient files.

The measured inputs (119 processing conditions, 17 extruded-Mg alloy classes)
are not distributed with this repository. This script writes files with the
same structure so that the texture-oversampling pipeline can be run end to end:

* one file per "condition": ``<Class>_extruded_<T>_<v>_odf_harmonics.txt``,
* 9,139 lines of ``real imag`` (MTEX SO3FunHarmonic coefficients, bandwidth 18),
* the per-class condition counts of the paper dataset (2 ... 18 per class).

Each class is a random smooth prototype (decaying coefficient magnitudes, the
constant term fixed at 1 like a normalised ODF) and each condition is the
prototype plus a class-specific low-rank variation. The values are NOT physical
textures; use them only to exercise the code.

    python examples/make_demo_data.py                 # -> data/odf_harmonics, data/stats_copula

With ``--copula`` (default on) it also writes one grain-statistics file per
condition, ``data/stats_copula/<Class>_extruded_<T>_<v>_copula_params.txt``,
holding the five Gaussian-copula parameters (mu, sigma of ln d; alpha, beta
of the aspect-ratio Beta law; rho), drawn so that the expected grain count of
a 300 x 300 x 1 RVE at 2 um lies inside the 250-1000 band.
"""
import argparse
from pathlib import Path

import numpy as np

N_COEFF = 9139                     # SO3FunHarmonic, bandwidth 18
CLASS_SIZES = [8, 8, 18, 4, 4, 4, 4, 4, 4, 9, 8, 9, 9, 11, 2, 11, 2]  # 119 conditions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/odf_harmonics"))
    ap.add_argument("--copula-out", type=Path, default=Path("data/stats_copula"))
    ap.add_argument("--no-copula", dest="copula", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    decay = 1.0 / (1.0 + np.arange(N_COEFF) / 150.0) ** 1.5
    n_files = 0
    if args.copula:
        args.copula_out.mkdir(parents=True, exist_ok=True)
    for k, n_cond in enumerate(CLASS_SIZES):
        mu_c, sig_c = rng.uniform(2.25, 2.55), rng.uniform(0.33, 0.45)       # class centre (ln um)
        name = f"Demo{chr(ord('A') + k)}_extruded"
        proto = rng.normal(size=(N_COEFF, 2)) * decay[:, None]
        basis = rng.normal(size=(3, N_COEFF, 2)) * decay[None, :, None]
        for j in range(n_cond):
            x = proto + 0.3 * np.tensordot(rng.normal(size=3), basis, axes=1)
            x += 0.02 * rng.normal(size=x.shape) * decay[:, None]
            x[0] = (1.0, 0.0)
            temperature, speed = 200 + 25 * j, [0.6, 1.0, 2.4][j % 3]   # extrusion T (degC), ram speed (mm/s)
            np.savetxt(args.out / f"{name}_{temperature}_{speed}_odf_harmonics.txt", x, fmt="%.6f")
            if args.copula:
                mu, sig = mu_c + rng.normal(0, 0.05), sig_c + rng.normal(0, 0.02)
                alpha, beta, rho = rng.uniform(4, 7), rng.uniform(2, 4), rng.uniform(0.0, 0.4)
                (args.copula_out / f"{name}_{temperature}_{speed}_copula_params.txt").write_text(
                    "# mu_log_esd sigma_log_esd alpha_ar beta_ar rho\n"
                    f"{mu}\n{sig}\n{alpha}\n{beta}\n{rho}\n")
            n_files += 1
    print(f"wrote {n_files} files ({len(CLASS_SIZES)} classes) to {args.out}")


if __name__ == "__main__":
    main()
