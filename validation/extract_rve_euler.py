"""Extract voxel Euler angles from .dream3d files in specified folders."""
import h5py
import numpy as np
import os
import sys

# usage: python validation/extract_rve_euler.py <folder-with-.dream3d> [...]
# Writes <name>_euler_deg.txt (one row per voxel: phi1 Phi phi2 in degrees) next to
# each .dream3d file, the input format of the MTEX pole-figure scripts in validation/.
folders = sys.argv[1:]
if not folders:
    sys.exit(__doc__ + "\nusage: extract_rve_euler.py <folder> [<folder> ...]")
BASE = ""

EULER_PATH = 'DataContainers/SyntheticVolumeDataContainer/CellData/EulerAngles'

total = 0
for folder in folders:
    folder_path = os.path.join(BASE, folder)
    dream3d_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.dream3d')])
    
    for fname in dream3d_files:
        fpath = os.path.join(folder_path, fname)
        out_txt = fpath.replace('.dream3d', '_euler_deg.txt')
        
        if os.path.exists(out_txt):
            continue
        
        try:
            with h5py.File(fpath, 'r') as f:
                euler = f[EULER_PATH][:]
            euler = euler.reshape(-1, 3)
            euler_deg = np.degrees(euler)
            np.savetxt(out_txt, euler_deg, fmt='%.6f')
            total += 1
            print(f'  [{total}] {folder}/{fname} -> {euler_deg.shape[0]} voxels')
        except Exception as e:
            print(f'  ERROR: {folder}/{fname}: {e}', file=sys.stderr)

print(f'\nDone. Extracted Euler angles for {total} files.')
