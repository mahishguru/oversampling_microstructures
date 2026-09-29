import os
import numpy as np
import pandas as pd
from scipy.stats import beta, gaussian_kde
import traceback

# User settings
ROOT_DIR = 'data/measured'
STATS_DIR = 'data/stats'
BETA_DIR = 'data/stats_aspect_beta'
SIGMA_CUTOFF_MIN = 3.0
SIGMA_CUTOFF_MAX = 3.0
BIN_STEP_SIZE = 1

os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(BETA_DIR, exist_ok=True)

def process_file(filepath):
    try:
        # Extract µm/pixel from filename: always the third-to-last underscore-separated value
        fname = os.path.basename(filepath)
        try:
            parts = fname.split('_')
            um_per_px = float(parts[2])
        except Exception:
            print(f"ERROR: Could not robustly parse µm/pixel from filename: {fname}")
            return False
        df = pd.read_csv(filepath)
        if 'Area' not in df.columns or 'Width' not in df.columns or 'Height' not in df.columns:
            print(f"ERROR: Missing required columns in {fname}")
            return False
        # ESD
        areas = df['Area'].values
        esd_px = 2 * np.sqrt(areas / np.pi)
        feature_esd = esd_px * um_per_px
        feature_esd = feature_esd[feature_esd > 0]
        # Log-normal fit
        log_esd = np.log(feature_esd)
        mu = np.mean(log_esd)
        sigma = np.std(log_esd)
        min_esd = np.exp(mu - SIGMA_CUTOFF_MIN * sigma)
        max_esd = np.exp(mu + SIGMA_CUTOFF_MAX * sigma)
        num_bins = int(np.floor((max_esd - min_esd) / BIN_STEP_SIZE)) + 1
        bin_edges = np.linspace(min_esd, max_esd, num_bins + 1)
        # Aspect ratio (B/A)
        df['Width_um'] = df['Width'] * um_per_px
        df['Height_um'] = df['Height'] * um_per_px
        df['A'] = np.maximum(df['Width_um'], df['Height_um'])
        df['B'] = np.minimum(df['Width_um'], df['Height_um'])
        df['aspect_ratio'] = df['B'] / df['A']
        eq_diameter_um = 2 * np.sqrt(df['Area'].values / np.pi) * um_per_px
        df['eq_diameter_um'] = eq_diameter_um
        df = df[df['eq_diameter_um'] > 0]
        df['diameter_bin'] = pd.cut(x=df['eq_diameter_um'], bins=bin_edges, right=True, include_lowest=False)
        df = df.dropna(subset=['diameter_bin', 'aspect_ratio'])
        # Aspect ratio statistics
        ar_vals = df['aspect_ratio'].dropna()
        aspect_mu = np.mean(ar_vals) if len(ar_vals) > 0 else 0.0
        aspect_sigma = np.std(ar_vals) if len(ar_vals) > 0 else 0.0
        # Beta fit per bin
        def beta_fit_stats(subdf):
            ar = subdf['aspect_ratio'].dropna()
            n = len(ar)
            if n < 5:
                return pd.Series({'grain_count': n, 'alpha': np.nan, 'beta': np.nan})
            ar = ar[(ar > 0) & (ar < 1)]
            if len(ar) < 2:
                return pd.Series({'grain_count': n, 'alpha': np.nan, 'beta': np.nan})
            try:
                a, b, loc, scale = beta.fit(ar, floc=0, fscale=1)
            except Exception:
                return pd.Series({'grain_count': n, 'alpha': np.nan, 'beta': np.nan})
            return pd.Series({'grain_count': n, 'alpha': a, 'beta': b})
        # Silence pandas DeprecationWarning by excluding grouping columns in groupby.apply (future-proof)
        binned_stats = df.groupby('diameter_bin', observed=True, group_keys=False).apply(beta_fit_stats, include_groups=False).reset_index()
        all_bins = pd.Categorical(pd.cut(df['eq_diameter_um'], bins=bin_edges, right=True, include_lowest=False).cat.categories, ordered=True)
        binned_stats = binned_stats.set_index('diameter_bin').reindex(all_bins).reset_index()
        binned_stats = binned_stats.rename(columns={'index': 'diameter_bin'})
        binned_stats['grain_count'] = binned_stats['grain_count'].fillna(0).astype(int)
        binned_stats['alpha'] = binned_stats['alpha'].fillna(0)
        binned_stats['beta'] = binned_stats['beta'].fillna(0)
        # KDE sampling for zero alpha/beta
        from scipy.stats import gaussian_kde
        nonzero_pairs = binned_stats[(binned_stats['alpha'] > 0) & (binned_stats['beta'] > 0)][['alpha', 'beta']].values
        unique_pairs = np.unique(nonzero_pairs, axis=0) if len(nonzero_pairs) > 0 else []
        used_kde = False
        if len(nonzero_pairs) > 1 and len(unique_pairs) > 2:
            try:
                kde = gaussian_kde(nonzero_pairs.T)
                used_kde = True
            except Exception:
                used_kde = False
        for idx, row in binned_stats.iterrows():
            if row['alpha'] == 0 or row['beta'] == 0:
                if used_kde:
                    for _ in range(100):
                        sampled = kde.resample(1).flatten()
                        if sampled[0] > 0 and sampled[1] > 0:
                            binned_stats.at[idx, 'alpha'] = sampled[0]
                            binned_stats.at[idx, 'beta'] = sampled[1]
                            break
                elif len(nonzero_pairs) > 0:
                    sampled_pair = nonzero_pairs[np.random.choice(len(nonzero_pairs))]
                    binned_stats.at[idx, 'alpha'] = sampled_pair[0]
                    binned_stats.at[idx, 'beta'] = sampled_pair[1]
        # Get material name from the filepath (first directory after ROOT_DIR)
        rel_path = os.path.relpath(filepath, ROOT_DIR)
        material_name = rel_path.split(os.sep)[0]
        # Save stats file with material name prefix
        base = os.path.basename(filepath)
        base_out = base.replace('_cropped_mask_statistics.csv', '').replace('_mask_statistics.csv', '')
        stats_path = os.path.join(STATS_DIR, f"{material_name}_{base_out}_stats.txt")
        with open(stats_path, 'w') as f:
            # Header: mu_lnESD sigma_lnESD aspect_mu aspect_sigma num_bins
            f.write("# mu_lnESD sigma_lnESD aspect_mu aspect_sigma num_bins\n")
            f.write(f"{mu}\n{sigma}\n{aspect_mu}\n{aspect_sigma}\n{num_bins}\n")
        # Save beta params file with material name prefix
        beta_path = os.path.join(BETA_DIR, f"{material_name}_{base_out}_stats_aspect_beta.txt")
        np.savetxt(beta_path, binned_stats[['alpha', 'beta']].values, fmt='%.6f')
        print(f"Processed: {filepath}\nSaved: {stats_path}\nSaved: {beta_path}")
        return True
    except Exception as e:
        print(f"ERROR processing {filepath}: {e}\n{traceback.format_exc()}")
        return False

def find_temp_strain_dirs(root):
    material_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    temp_strain_dirs = []
    for mat_dir in material_dirs:
        for d in os.listdir(mat_dir):
            ts_dir = os.path.join(mat_dir, d)
            if os.path.isdir(ts_dir):
                temp_strain_dirs.append(ts_dir)
    return temp_strain_dirs

def collect_csvs_in_dir(ts_dir):
    csvs = []
    for dirpath, dirnames, filenames in os.walk(ts_dir):
        for filename in filenames:
            if filename.endswith('_mask_statistics.csv'):
                csvs.append(os.path.join(dirpath, filename))
    return csvs

def process_temp_strain(ts_dir, stats_dir, beta_dir):
    csv_files = collect_csvs_in_dir(ts_dir)
    if not csv_files:
        print(f"No CSVs found in {ts_dir}")
        return False
    all_grains = []
    for filepath in csv_files:
        fname = os.path.basename(filepath)
        try:
            parts = fname.split('_')
            um_per_px = float(parts[2])
        except Exception:
            print(f"ERROR: Could not parse µm/pixel from filename: {fname}")
            continue
        df = pd.read_csv(filepath)
        if 'Area' not in df.columns or 'Width' not in df.columns or 'Height' not in df.columns:
            print(f"ERROR: Missing required columns in {fname}")
            continue
        df['um_per_px'] = um_per_px
        all_grains.append(df)
    if not all_grains:
        print(f"No valid grains in {ts_dir}")
        return False
    df = pd.concat(all_grains, ignore_index=True)
    # ESD and conversion
    areas = df['Area'].values
    esd_px = 2 * np.sqrt(areas / np.pi)
    feature_esd = esd_px * df['um_per_px'].values
    feature_esd = feature_esd[feature_esd > 0]
    log_esd = np.log(feature_esd)
    mu = np.mean(log_esd)
    sigma = np.std(log_esd)
    min_esd = np.exp(mu - SIGMA_CUTOFF_MIN * sigma)
    max_esd = np.exp(mu + SIGMA_CUTOFF_MAX * sigma)
    num_bins = int(np.floor((max_esd - min_esd) / BIN_STEP_SIZE)) + 1
    bin_edges = np.linspace(min_esd, max_esd, num_bins + 1)
    # Aspect ratio (B/A)
    df['Width_um'] = df['Width'] * df['um_per_px']
    df['Height_um'] = df['Height'] * df['um_per_px']
    df['A'] = np.maximum(df['Width_um'], df['Height_um'])
    df['B'] = np.minimum(df['Width_um'], df['Height_um'])
    df['aspect_ratio'] = df['B'] / df['A']
    eq_diameter_um = 2 * np.sqrt(df['Area'].values / np.pi) * df['um_per_px']
    df['eq_diameter_um'] = eq_diameter_um
    df = df[df['eq_diameter_um'] > 0]
    df['diameter_bin'] = pd.cut(x=df['eq_diameter_um'], bins=bin_edges, right=True, include_lowest=False)
    df = df.dropna(subset=['diameter_bin', 'aspect_ratio'])
    ar_vals = df['aspect_ratio'].dropna()
    aspect_mu = np.mean(ar_vals) if len(ar_vals) > 0 else 0.0
    aspect_sigma = np.std(ar_vals) if len(ar_vals) > 0 else 0.0
    def beta_fit_stats(subdf):
        ar = subdf['aspect_ratio'].dropna()
        n = len(ar)
        if n < 5:
            return pd.Series({'grain_count': n, 'alpha': np.nan, 'beta': np.nan})
        ar = ar[(ar > 0) & (ar < 1)]
        if len(ar) < 2:
            return pd.Series({'grain_count': n, 'alpha': np.nan, 'beta': np.nan})
        try:
            a, b, loc, scale = beta.fit(ar, floc=0, fscale=1)
        except Exception:
            return pd.Series({'grain_count': n, 'alpha': np.nan, 'beta': np.nan})
        return pd.Series({'grain_count': n, 'alpha': a, 'beta': b})
    binned_stats = df.groupby('diameter_bin', observed=True, group_keys=False).apply(beta_fit_stats, include_groups=False).reset_index()
    all_bins = pd.Categorical(pd.cut(df['eq_diameter_um'], bins=bin_edges, right=True, include_lowest=False).cat.categories, ordered=True)
    binned_stats = binned_stats.set_index('diameter_bin').reindex(all_bins).reset_index()
    binned_stats = binned_stats.rename(columns={'index': 'diameter_bin'})
    binned_stats['grain_count'] = binned_stats['grain_count'].fillna(0).astype(int)
    binned_stats['alpha'] = binned_stats['alpha'].fillna(0)
    binned_stats['beta'] = binned_stats['beta'].fillna(0)
    nonzero_pairs = binned_stats[(binned_stats['alpha'] > 0) & (binned_stats['beta'] > 0)][['alpha', 'beta']].values
    unique_pairs = np.unique(nonzero_pairs, axis=0) if len(nonzero_pairs) > 0 else []
    used_kde = False
    if len(nonzero_pairs) > 1 and len(unique_pairs) > 2:
        try:
            kde = gaussian_kde(nonzero_pairs.T)
            used_kde = True
        except Exception:
            used_kde = False
    for idx, row in binned_stats.iterrows():
        if row['alpha'] == 0 or row['beta'] == 0:
            if used_kde:
                for _ in range(100):
                    sampled = kde.resample(1).flatten()
                    if sampled[0] > 0 and sampled[1] > 0:
                        binned_stats.at[idx, 'alpha'] = sampled[0]
                        binned_stats.at[idx, 'beta'] = sampled[1]
                        break
            elif len(nonzero_pairs) > 0:
                sampled_pair = nonzero_pairs[np.random.choice(len(nonzero_pairs))]
                binned_stats.at[idx, 'alpha'] = sampled_pair[0]
                binned_stats.at[idx, 'beta'] = sampled_pair[1]
    # Save files
    rel_path = os.path.relpath(ts_dir, ROOT_DIR)
    material_name, temp_strain = rel_path.split(os.sep)[:2]
    stats_path = os.path.join(stats_dir, f"{material_name}_{temp_strain}_stats.txt")
    with open(stats_path, 'w') as f:
        # Header: mu_lnESD sigma_lnESD aspect_mu aspect_sigma num_bins
        f.write("# mu_lnESD sigma_lnESD aspect_mu aspect_sigma num_bins\n")
        f.write(f"{mu}\n{sigma}\n{aspect_mu}\n{aspect_sigma}\n{num_bins}\n")
    beta_path = os.path.join(beta_dir, f"{material_name}_{temp_strain}_stats_aspect_beta.txt")
    np.savetxt(beta_path, binned_stats[['alpha', 'beta']].values, fmt='%.6f')
    print(f"Processed: {ts_dir}\nSaved: {stats_path}\nSaved: {beta_path}")
    return True

def find_all_csv_files(root):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('_mask_statistics.csv'):
                matches.append(os.path.join(dirpath, filename))
    return matches

def find_temp_strain_dirs(root):
    material_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    temp_strain_dirs = []
    for mat_dir in material_dirs:
        for d in os.listdir(mat_dir):
            ts_dir = os.path.join(mat_dir, d)
            if os.path.isdir(ts_dir):
                temp_strain_dirs.append(ts_dir)
    return temp_strain_dirs

def collect_csvs_in_dir(ts_dir):
    csvs = []
    for dirpath, dirnames, filenames in os.walk(ts_dir):
        for filename in filenames:
            if filename.endswith('_mask_statistics.csv'):
                csvs.append(os.path.join(dirpath, filename))
    return csvs

def main():
    temp_strain_dirs = find_temp_strain_dirs(ROOT_DIR)
    for ts_dir in temp_strain_dirs:
        ok = process_temp_strain(ts_dir, STATS_DIR, BETA_DIR)
        if not ok:
            print(f"Failed: {ts_dir}")

if __name__ == '__main__':
    main()
