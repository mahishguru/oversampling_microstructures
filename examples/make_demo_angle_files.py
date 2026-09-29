#!/usr/bin/env python3
"""Write DREAM.3D angle files without MATLAB (demo / smoke test only).

In the real pipeline, mtex/SHcoeffs_to_Dream3d_odf_batch_parallel.m reconstructs
each synthetic ODF from its GSH coefficients and draws 100,000 orientations from
it. This script writes files with the same name and format for every synthetic
sample, filled with orientations scattered around a basal fibre, so that
rve_synthesis/04_generate_rves.py can be tried end to end:

    <odf-root>/<Class>/dream3d_angles_uniform/<Class>_<i>_dream3d_odf_angles.txt

    python examples/make_demo_angle_files.py --odf-root data/odf_harmonics_sampled --n 20000
"""
import argparse
import re
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--odf-root", type=Path, default=Path("data/odf_harmonics_sampled"))
    ap.add_argument("--n", type=int, default=20000, help="orientations per file (paper: 100,000)")
    ap.add_argument("--limit", type=int, default=0, help="files per class (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_written = 0
    for cls in sorted(p for p in args.odf_root.iterdir() if p.is_dir()):
        ids = sorted(int(m.group(1)) for f in cls.glob("*.txt")
                     if (m := re.fullmatch(rf"{re.escape(cls.name)}_(\d+)\.txt", f.name)))
        if args.limit:
            ids = ids[: args.limit]
        out_dir = cls / "dream3d_angles_uniform"
        out_dir.mkdir(exist_ok=True)
        for i in ids:
            phi1 = rng.uniform(0, 360, args.n)                   # degrees, Bunge ZXZ
            Phi = np.clip(np.abs(rng.normal(0, 20, args.n)), 0, 180)
            phi2 = rng.uniform(0, 60, args.n)
            data = np.column_stack([phi1, Phi, phi2, np.ones(args.n), np.ones(args.n)])
            with open(out_dir / f"{cls.name}_{i}_dream3d_odf_angles.txt", "w") as fh:
                fh.write("# DREAM.3D StatsGenerator Angles Input File\n")
                fh.write("# Demo orientations (basal fibre), not reconstructed from GSH coefficients\n")
                fh.write("# Euler0 Euler1 Euler2 Weight Sigma\n")
                fh.write(f"Angle Count:{args.n}\n")
                np.savetxt(fh, data, fmt="%.6f")
            n_written += 1
    print(f"wrote {n_written} angle files under {args.odf_root}")


if __name__ == "__main__":
    main()
