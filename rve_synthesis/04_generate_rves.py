import json
import copy
import os
import math
import random
import subprocess
import re
import signal
import multiprocessing as mp
import argparse
import sys
import shutil
import threading

# =========================
# CONSTANT KEYS
# =========================
kBinNumbers = "BinNumber"
kAlphaCOverA = "Alpha"
kBetaCOverA = "Beta"
kAlphaOmega3 = "Alpha"
kBetaOmega3 = "Beta"
kMu = "Average"
kSigma = "Standard Deviation"

# =========================
# USER PARAMETERS
# =========================
min_cutoff = 3
max_cutoff = 3

# Per-sample processing timeout (seconds); samples that exceed this are skipped.
_PROCESS_TIMEOUT_SECONDS = 3600

# MatchCrystallography settings (validated optimal values)
_MAX_ITERATIONS = 500000   # 5x default; improves ODF matching convergence
_ODF_SIGMA = 1.0           # sigma < 1 collapses texture; > 1 over-smooths

# Crystal symmetry for the (hexagonal Mg) phase. In DREAM.3D/EbsdLib the enum is
# Hexagonal_High=0, Cubic_High=1. All alloys here are HCP Mg, so this MUST be 0.
# Setting it to 1 (cubic) folds the hex ODF into the wrong fundamental zone and
# scatters the texture -> realized RVE collapses to near-random. Forced here so a
# stale template can never reintroduce the bug.
_CRYSTAL_SYMMETRY = 0      # 0 = Hexagonal_High (HCP); do NOT set to 1 (cubic)

# Parallel + resume behavior
num_workers = 20
skip_if_done = True  # skip samples that already have final outputs
require_xdmf = False  # only used for deciding what "done" means
maxtasksperchild = 500  # recycle workers periodically to limit memory drift

# =========================
# PATHS
# =========================
template_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.json")

# Input roots (grain statistics from grain_statistics/03*, angle files from
# mtex/SHcoeffs_to_Dream3d_odf_batch_parallel.m); override via environment variables.
stats_root = os.environ.get(
    "RVE_STATS_ROOT",
    "data/stats_combined_sampled",
)
odf_root = os.environ.get(
    "RVE_ODF_ROOT",
    "data/odf_harmonics_sampled",
)

rve_root = os.environ.get("RVE_OUT_ROOT", "data/rve")
# DREAM.3D's PipelineRunner resolves paths relative to its own location: use absolute paths.
stats_root, odf_root, rve_root = (os.path.abspath(p) for p in (stats_root, odf_root, rve_root))

# DREAM.3D 6.5.171 (SIMPL) PipelineRunner binary
pipeline_runner = os.environ.get("DREAM3D_PIPELINE_RUNNER", "PipelineRunner")

os.makedirs(rve_root, exist_ok=True)

# =========================
# BIN FUNCTION
# =========================
def determine_bin_numbers(max_val, min_val, bin_step_size, bin_numbers):
    for i in range(len(bin_numbers)):
        bin_numbers[i] = min_val + i * bin_step_size

# =========================
# DREAM3D FUNCTIONS (UNCHANGED)
# =========================
def initialize_c_over_a_table_model(data, aspect_ratio2):
    count = len(data[kBinNumbers])
    alphas, betas = [], []

    for i in range(count):
        alpha = (1.1 + (28.9 / aspect_ratio2)) + random.random()
        beta = (30 - (28.9 / aspect_ratio2)) + random.random()
        alphas.append(alpha)
        betas.append(beta)

    data[kAlphaCOverA] = alphas
    data[kBetaCOverA] = betas


def initialize_neighbor_table_model(data):
    count = len(data[kBinNumbers])
    mus, sigmas = [], []
    middle_bin = count // 2

    for i in range(count):
        mu = math.log(max(1e-10, 8.0 + (i - middle_bin)))
        sigma = 0.3 + ((middle_bin - i) / (middle_bin * 10))
        mus.append(mu)
        sigmas.append(sigma)

    data[kMu] = mus
    data[kSigma] = sigmas


def initialize_omega3_table_model(data):
    count = len(data[kBinNumbers])
    alphas, betas = [], []

    for i in range(count):
        alpha = 10.0 + random.random()
        beta = 1.5 + (0.5 * random.random())
        alphas.append(alpha)
        betas.append(beta)

    data[kAlphaOmega3] = alphas
    data[kBetaOmega3] = betas


# =========================
# HELPERS
# =========================
def _list_subdirs(path):
    try:
        names = os.listdir(path)
    except FileNotFoundError:
        return []
    subdirs = []
    for name in names:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            subdirs.append(name)
    return sorted(subdirs)


def _collect_sample_files(material, stats_dir, odf_dir):
    stats_map = {}
    aspect_map = {}
    odf_map = {}

    stats_re = re.compile(rf"^{re.escape(material)}_(\d+)_stats\.txt$")
    aspect_re = re.compile(rf"^{re.escape(material)}_(\d+)_stats_aspect.*\.txt$")
    angle_re = re.compile(rf"^{re.escape(material)}_(\d+)_dream3d_odf_angles\.txt$")

    for fname in os.listdir(stats_dir):
        m = stats_re.match(fname)
        if m:
            stats_map[int(m.group(1))] = os.path.join(stats_dir, fname)
            continue
        m = aspect_re.match(fname)
        if m:
            aspect_map[int(m.group(1))] = os.path.join(stats_dir, fname)

    # Angle files are in <odf_dir>/dream3d_angles_uniform/
    angle_dir = os.path.join(odf_dir, "dream3d_angles_uniform")
    if os.path.isdir(angle_dir):
        for fname in os.listdir(angle_dir):
            m = angle_re.match(fname)
            if m:
                odf_map[int(m.group(1))] = os.path.join(angle_dir, fname)

    common_ids = sorted(set(stats_map) & set(aspect_map) & set(odf_map))
    return common_ids, stats_map, aspect_map, odf_map


def _parse_dream3d_angle_file(filepath):
    """Parse DREAM.3D StatsGenerator angle file.

    Format: header lines starting with # or 'Angle Count:N',
    then rows of: phi1 Phi phi2 Weight Sigma (degrees).

    Returns (euler1_rad, euler2_rad, euler3_rad, weights, sigmas).
    """
    phi1, Phi, phi2, weights, sigmas = [], [], [], [], []
    reading = False
    deg2rad = math.pi / 180.0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Angle Count:"):
                reading = True
                continue
            if reading:
                parts = line.split()
                if len(parts) >= 5:
                    phi1.append(float(parts[0]) * deg2rad)
                    Phi.append(float(parts[1]) * deg2rad)
                    phi2.append(float(parts[2]) * deg2rad)
                    weights.append(float(parts[3]))
                    sigmas.append(float(parts[4]))

    return phi1, Phi, phi2, weights, sigmas


def _set_pipeline_output_file(pipeline_json, dream3d_file):
    # Most templates use filter index "7" with OutputFile.
    if isinstance(pipeline_json.get("7"), dict) and "OutputFile" in pipeline_json["7"]:
        pipeline_json["7"]["OutputFile"] = dream3d_file
        return True

    # Fallback: search for DataContainerWriter filter.
    for _, node in pipeline_json.items():
        if not isinstance(node, dict):
            continue
        if node.get("Filter_Name") == "DataContainerWriter" and "OutputFile" in node:
            node["OutputFile"] = dream3d_file
            return True
    return False


def _output_paths(material: str, sample_id: int, out_dir: str):
    base = os.path.join(out_dir, f"{material}_{sample_id}")
    return {
        "json": base + ".json",
        "dream3d": base + ".dream3d",
        "xdmf": base + ".xdmf",
        "vti": base + ".vti",
        "png": base + ".png",
    }



def _exists_nonempty(path: str, min_bytes: int = 1) -> bool:
    try:
        return os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _is_done(paths: dict) -> bool:
    return _exists_nonempty(paths["dream3d"])


def _needs_damask_only(paths: dict) -> bool:
    return _exists_nonempty(paths["dream3d"]) and (not _exists_nonempty(paths["vti"]))


# Global ref set by pool initializer so damask is imported once per worker process
_damask_mod = None


def _worker_init():
    """Called once when each pool worker process starts.
    Pre-imports the heavy `damask` module so it is ready for every task."""
    global _damask_mod
    try:
        import damask
        _damask_mod = damask
    except Exception:
        _damask_mod = None


def _damask_export(dream3d_path: str, vti_path: str):
    global _damask_mod
    if _damask_mod is None:
        import damask
        _damask_mod = damask

    t = _damask_mod.GeomGrid.load_DREAM3D(
        fname=dream3d_path,
        feature_IDs="FeatureIds",
        cell_data="CellData",
        phases="Phases",
        Euler_angles="EulerAngles",
        base_group="DataContainers/SyntheticVolumeDataContainer",
    )
    t.save(fname=vti_path, compress=True)


def _save_grain_png(dream3d_path: str, png_path: str):
    """Save a 2D grain map PNG from the DREAM.3D file (first z-slice)."""
    import h5py
    import numpy as np

    with h5py.File(dream3d_path, "r") as f:
        fids = f["DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds"][:]

    # Squeeze to 2D (NZ=1)
    fids = fids.squeeze()
    if fids.ndim == 3:
        fids = fids[0]  # take first z-slice

    # Use matplotlib with Agg backend (no display needed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(fids, cmap="nipy_spectral", interpolation="nearest", origin="lower")
    ax.set_axis_off()
    ax.set_title(f"Grains: {int(fids.max())}", fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# =========================
# LOAD TEMPLATE
# =========================
with open(template_json, "r") as f:
    base_data = json.load(f)

json_base = os.path.splitext(os.path.basename(template_json))[0]

base_stats = base_data["0"]["StatsDataArray"]["1"]


def _process_one_sample(task):
    """Worker: generates JSON, runs DREAM3D, then exports DAMASK geometry."""
    material = task["material"]
    sample_id = task["sample_id"]
    out_dir = task["out_dir"]

    paths = task["paths"]
    mode = task.get("mode", "full")  # "full" | "damask_only"

    os.makedirs(out_dir, exist_ok=True)

    # Fast-path: skip completed samples
    if skip_if_done and _is_done(paths):
        return {
            "material": material,
            "sample_id": sample_id,
            "ok": True,
            "stage": "skipped",
            "out_dir": out_dir,
        }

    # --- 2-minute per-sample timeout ---
    def _on_timeout(signum, frame):
        raise TimeoutError(f"Timed out after {_PROCESS_TIMEOUT_SECONDS}s")

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(_PROCESS_TIMEOUT_SECONDS)
    try:
        # Full pipeline inputs (also used for fallback if damask-only fails)
        stats_path = task["stats_path"]
        aspect_path = task["aspect_path"]
        odf_path = task["odf_path"]

        # Fast-path: if DREAM3D exists but VTI missing, only run DAMASK export
        if mode == "damask_only":
            try:
                if not _exists_nonempty(paths["dream3d"]):
                    raise FileNotFoundError(f"Dream3D file not found or empty: {paths['dream3d']}")
                _damask_export(paths["dream3d"], paths["vti"])
                return {
                    "material": material,
                    "sample_id": sample_id,
                    "ok": True,
                    "stage": "done_damask_only",
                    "out_dir": out_dir,
                }
            except TimeoutError:
                raise
            except Exception:
                # If the existing dream3d is corrupted/incomplete, fall back to full regeneration.
                mode = "full"

        # READ STATS FILE (SAFE)
        lines = []
        with open(stats_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    lines.append(float(line))
                except Exception:
                    continue

        if len(lines) < 5:
            return {
                "material": material,
                "sample_id": sample_id,
                "ok": False,
                "stage": "read_stats",
                "error": f"Invalid stats file: {stats_path}",
            }

        mu = lines[0]
        sigma = lines[1]
        bin_count = int(lines[4])

        if bin_count < 2:
            return {
                "material": material,
                "sample_id": sample_id,
                "ok": False,
                "stage": "read_stats",
                "error": f"Invalid bin_count={bin_count} in stats file: {stats_path}",
            }

        # BIN CALCULATION
        min_esd = math.exp(mu - min_cutoff * sigma)
        max_esd = math.exp(mu + max_cutoff * sigma)
        bin_step_size = (max_esd - min_esd) / (bin_count - 1)
        bin_numbers = [0.0] * bin_count
        determine_bin_numbers(max_esd, min_esd, bin_step_size, bin_numbers)

        # READ ASPECT FILE (B/A)
        alpha_list = []
        beta_list = []
        with open(aspect_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        alpha_list.append(float(parts[0]))
                        beta_list.append(float(parts[1]))
                    except Exception:
                        continue
        alpha_list = alpha_list[:bin_count]
        beta_list = beta_list[:bin_count]

        # PARSE DREAM.3D ANGLE FILE (validated approach: replaces entire ODF-Weights)
        try:
            euler1, euler2, euler3, odf_weights, odf_sigmas = _parse_dream3d_angle_file(odf_path)
        except Exception as e:
            return {
                "material": material,
                "sample_id": sample_id,
                "ok": False,
                "stage": "parse_odf",
                "error": f"Failed to parse angle file {odf_path}: {e}",
            }
        if len(odf_weights) < 10:
            return {
                "material": material,
                "sample_id": sample_id,
                "ok": False,
                "stage": "parse_odf",
                "error": f"Too few angle entries ({len(odf_weights)}) in {odf_path}",
            }

        # MODIFY JSON
        data = copy.deepcopy(base_data)
        stats = data["0"]["StatsDataArray"]["1"]
        # Force hexagonal crystal symmetry (HCP Mg). Guards against a stale
        # template that still carries the cubic (=1) bug.
        stats["Crystal Symmetry"] = _CRYSTAL_SYMMETRY
        stats["FeatureSize Distribution"]["Average"] = mu
        stats["FeatureSize Distribution"]["Standard Deviation"] = sigma
        stats["Bin Count"] = bin_count
        stats["Feature_Diameter_Info"] = [bin_step_size, max_esd, min_esd]
        stats["BinNumber"] = bin_numbers
        stats["FeatureSize Vs B Over A Distributions"]["Alpha"] = alpha_list
        stats["FeatureSize Vs B Over A Distributions"]["Beta"] = beta_list

        temp_data = {kBinNumbers: bin_numbers}
        aspect_ratio2 = 1 / 0.9
        initialize_c_over_a_table_model(temp_data, aspect_ratio2)
        initialize_neighbor_table_model(temp_data)
        initialize_omega3_table_model(temp_data)
        stats["FeatureSize Vs C Over A Distributions"]["Alpha"] = temp_data[kAlphaCOverA]
        stats["FeatureSize Vs C Over A Distributions"]["Beta"] = temp_data[kBetaCOverA]
        stats["FeatureSize Vs Neighbors Distributions"]["Average"] = temp_data[kMu]
        stats["FeatureSize Vs Neighbors Distributions"]["Standard Deviation"] = temp_data[kSigma]
        stats["FeatureSize Vs Omega3 Distributions"]["Alpha"] = temp_data[kAlphaOmega3]
        stats["FeatureSize Vs Omega3 Distributions"]["Beta"] = temp_data[kBetaOmega3]
        # Replace entire ODF-Weights with angle file data (validated approach)
        stats["ODF-Weights"]["Euler 1"] = euler1
        stats["ODF-Weights"]["Euler 2"] = euler2
        stats["ODF-Weights"]["Euler 3"] = euler3
        stats["ODF-Weights"]["Weight"] = odf_weights
        stats["ODF-Weights"]["Sigma"] = [_ODF_SIGMA] * len(odf_weights)

        # Set MatchCrystallography MaxIterations (filter "5")
        if "5" in data and "MaxIterations" in data["5"]:
            data["5"]["MaxIterations"] = _MAX_ITERATIONS

        if not _set_pipeline_output_file(data, paths["dream3d"]):
            # keep going; DREAM3D will fail if output isn't set
            pass

        with open(paths["json"], "w") as f:
            json.dump(data, f, indent=4)

        # RUN DREAM3D
        result = subprocess.run(
            [pipeline_runner, "-p", paths["json"]],
            capture_output=True,
            text=True,
            timeout=_PROCESS_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return {
                "material": material,
                "sample_id": sample_id,
                "ok": False,
                "stage": "dream3d",
                "error": result.stderr.strip() or "Dream3D failed",
            }

        # Delete the pipeline JSON to save disk space (keep xdmf for ParaView).
        try:
            if os.path.exists(paths["json"]):
                os.remove(paths["json"])
        except OSError:
            pass

        # DAMASK GEOMETRY
        try:
            # subprocess.run already waited for PipelineRunner to finish;
            # no sleep needed — the file is flushed to disk.
            _damask_export(paths["dream3d"], paths["vti"])
        except TimeoutError:
            raise
        except Exception as e:
            return {
                "material": material,
                "sample_id": sample_id,
                "ok": False,
                "stage": "damask",
                "error": str(e),
            }

        # PNG grain map
        try:
            _save_grain_png(paths["dream3d"], paths["png"])
        except Exception:
            pass  # non-critical; don't fail the sample over a PNG

        return {
        "material": material,
        "sample_id": sample_id,
        "ok": True,
            "stage": "done",
            "out_dir": out_dir,
        }
    except (TimeoutError, subprocess.TimeoutExpired):
        # Remove any partially written output files so they are not mistaken
        # for completed outputs on the next run.
        for p in (paths["dream3d"], paths["vti"], paths["xdmf"], paths["json"], paths["png"]):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        return {
            "material": material,
            "sample_id": sample_id,
            "ok": False,
            "stage": "timeout",
            "error": f"Skipped: exceeded {_PROCESS_TIMEOUT_SECONDS}s timeout",
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _parse_args():
    p = argparse.ArgumentParser(description="Batch generate RVEs using DREAM3D + DAMASK export.")
    p.add_argument(
        "--materials",
        nargs="*",
        default=None,
        help="Optional list of material class folders to run (default: all).",
    )
    p.add_argument(
        "--limit-per-material",
        type=int,
        default=None,
        help="Optional limit of samples per material (for testing/resume validation).",
    )
    p.add_argument("--workers", type=int, default=num_workers, help="Number of parallel worker processes.")
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable skipping already-generated samples.",
    )
    p.add_argument(
        "--require-xdmf",
        action="store_true",
        help="Treat a sample as done only if .xdmf exists too.",
    )
    p.add_argument(
        "--local-staging",
        type=str,
        default=None,
        help="Write outputs to this local (fast) directory, then transfer to rve_root in background.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of samples per batch when using --local-staging (default: 5000).",
    )
    p.add_argument(
        "--stage-inputs-dir",
        type=str,
        default=None,
        help=(
            "Pre-copy each batch's input files (stats/aspect/ODF) into this local "
            "(fast) directory with a single sequential reader before generating, so "
            "the 20 workers read inputs from local SSD instead of thrashing the slow "
            "USB with concurrent random reads. Cleared after each batch."
        ),
    )
    p.add_argument(
        "--no-transfer",
        action="store_true",
        help=(
            "Do NOT transfer outputs to the USB rve_root. Keep everything in the "
            "--local-staging dir (SSD). Used by the SSD->PNG pipeline, where a "
            "separate per-class encoder converts the .dream3d to PNG locally and "
            "ships only the tiny PNGs to the USB. With this flag the resume "
            "skip-scan looks ONLY at the local staging dir (not the USB), so "
            "samples that only exist on the USB are regenerated to the SSD."
        ),
    )
    return p.parse_args()


def _transfer_and_cleanup(local_staging, final_root):
    """Rsync local staging dir to final root, then remove local copies."""
    print(f"\n📦 Transferring batch from {local_staging} → {final_root} ...")
    try:
        subprocess.run(
            ["rsync", "-a", "--remove-source-files", local_staging + "/", final_root + "/"],
            check=True,
            timeout=7200,
        )
        # Remove empty dirs left behind by --remove-source-files
        subprocess.run(["find", local_staging, "-type", "d", "-empty", "-delete"], check=False)
        print(f"✅ Transfer complete.")
    except Exception as e:
        print(f"⚠️ Transfer error: {e}. Files remain in {local_staging}")


def _transfer_in_background(local_staging, final_root):
    """Start transfer in a background thread, return the thread."""
    t = threading.Thread(target=_transfer_and_cleanup, args=(local_staging, final_root), daemon=True)
    t.start()
    return t


def _stage_batch_inputs(batch_tasks, cache_dir):
    """Sequentially copy this batch's input files (stats/aspect/ODF) into a local
    cache directory and rewrite the task dicts to point at the local copies.

    A single sequential reader pulls each file from the slow USB once, instead of
    20 workers issuing concurrent random reads (which stall in D state on a HDD).
    Returns the list of local files created, for cleanup after the batch.
    """
    os.makedirs(cache_dir, exist_ok=True)
    staged = []
    for task in batch_tasks:
        for key in ("stats_path", "aspect_path", "odf_path"):
            src = task.get(key)
            if not src:
                continue
            # Basenames already encode material + sample id, so they are unique.
            dst = os.path.join(cache_dir, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    shutil.copyfile(src, dst)
                except FileNotFoundError:
                    # Leave the original path in place if the source is missing;
                    # the worker will report the failure for this sample.
                    continue
            task[key] = dst
            staged.append(dst)
    return staged


def _cleanup_staged_inputs(staged):
    """Delete the local input copies created for a finished batch."""
    for p in staged:
        try:
            os.remove(p)
        except OSError:
            pass


def main():
    global num_workers, skip_if_done, require_xdmf

    args = _parse_args()
    num_workers = max(1, int(args.workers))
    skip_if_done = (not args.no_skip)
    require_xdmf = bool(args.require_xdmf)

    material_classes = _list_subdirs(stats_root)
    if not material_classes:
        raise FileNotFoundError(f"No material-class subfolders found under: {stats_root}")

    if args.materials:
        wanted = set()
        for item in args.materials:
            if not item:
                continue
            for token in str(item).split(","):
                token = token.strip()
                if token:
                    wanted.add(token)
        material_classes = [m for m in material_classes if m in wanted]

    # Determine output root (local staging or direct to external)
    local_staging = args.local_staging
    batch_size = args.batch_size
    stage_inputs_dir = args.stage_inputs_dir
    no_transfer = args.no_transfer
    if stage_inputs_dir:
        os.makedirs(stage_inputs_dir, exist_ok=True)
        print(f"📥 Staging batch inputs locally: {stage_inputs_dir}")
    if local_staging:
        effective_rve_root = local_staging
        os.makedirs(local_staging, exist_ok=True)
        print(f"📁 Local staging: {local_staging} (batch size: {batch_size})")
    else:
        effective_rve_root = rve_root

    tasks = []
    skipped = 0
    for material in material_classes:
        stats_dir = os.path.join(stats_root, material)
        odf_dir = os.path.join(odf_root, material)
        if not os.path.isdir(odf_dir):
            print(f"⚠️ Skipping {material}: missing ODF folder: {odf_dir}")
            continue

        material_out_dir = os.path.join(effective_rve_root, material)
        os.makedirs(material_out_dir, exist_ok=True)
        # Final destination dir on the USB. Only created/used when we actually
        # transfer there. In --no-transfer mode we never touch the (slow) USB,
        # so skip this makedirs entirely.
        final_material_out_dir = os.path.join(rve_root, material)
        if not no_transfer:
            os.makedirs(final_material_out_dir, exist_ok=True)

        sample_ids, stats_map, aspect_map, odf_map = _collect_sample_files(material, stats_dir, odf_dir)
        if not sample_ids:
            print(f"⚠️ Skipping {material}: no matched (stats+aspect+ODF) sample IDs")
            continue

        print(f"\n=== Material: {material} | Matched samples: {len(sample_ids)} ===")
        if args.limit_per_material is not None:
            sample_ids = sample_ids[: max(0, int(args.limit_per_material))]

        # Build the set of already-done sample IDs with ONE directory scan per
        # location, instead of one os.path.getsize() per candidate sample. On a
        # slow USB drive, 100k+ individual stat calls take ~10 min per launch; a
        # single scandir (only the completed sample subdirs exist) is near-instant.
        # A sample is "done" if its .dream3d exists and is non-empty.
        #
        # Scan BOTH the final USB destination AND the local staging dir: if a
        # previous run was interrupted, finished RVEs may still be waiting in the
        # local stage (not yet rsync'd to USB). Counting them here means we never
        # regenerate already-finished samples on resume, and we don't have to
        # flush the stage to USB before resuming (the background per-batch
        # transfer drains it concurrently).
        done_ids = set()
        if skip_if_done:
            if no_transfer:
                # SSD->PNG pipeline: outputs never go to the USB, so "done" means
                # the .dream3d exists in the local staging dir. Do NOT count USB
                # copies as done (those get regenerated to the SSD so the encoder
                # has a single local source for the whole class).
                scan_dirs = [material_out_dir]
            else:
                scan_dirs = [final_material_out_dir]
                if material_out_dir != final_material_out_dir:
                    scan_dirs.append(material_out_dir)
            for scan_dir in scan_dirs:
                try:
                    with os.scandir(scan_dir) as it:
                        for entry in it:
                            if not entry.is_dir():
                                continue
                            d3 = os.path.join(entry.path, f"{material}_{entry.name}.dream3d")
                            if _exists_nonempty(d3):
                                done_ids.add(entry.name)
                except FileNotFoundError:
                    pass

        for sample_id in sample_ids:
            # Check done status against FINAL destination (external drive)
            if skip_if_done and str(sample_id) in done_ids:
                skipped += 1
                continue

            # Actual output goes to effective root (local staging or direct)
            out_dir = os.path.join(material_out_dir, str(sample_id))
            paths = _output_paths(material, sample_id, out_dir)

            mode = "full"
            if _needs_damask_only(paths):
                mode = "damask_only"

            tasks.append(
                {
                    "material": material,
                    "sample_id": sample_id,
                    "stats_path": stats_map[sample_id],
                    "aspect_path": aspect_map[sample_id],
                    "odf_path": odf_map[sample_id],
                    "out_dir": out_dir,
                    "paths": paths,
                    "mode": mode,
                }
            )

    # Interleave tasks round-robin by material so workers stay spread across
    # different material classes rather than all piling onto the same one.
    # Each material's list is reversed so we start from the end of the sample
    # list (higher IDs), which tend to process faster than the early ones.
    tasks_by_material: dict = {}
    for t in tasks:
        tasks_by_material.setdefault(t["material"], []).append(t)
    interleaved = []
    buckets = [list(reversed(v)) for v in tasks_by_material.values()]
    while any(buckets):
        for bucket in buckets:
            if bucket:
                interleaved.append(bucket.pop(0))
    tasks = interleaved

    total = 0
    failed = 0
    timed_out = 0
    done_damask_only = 0
    done_skipped_in_worker = 0

    if skipped and skip_if_done:
        print(f"\n⏭️  Pre-skip (already done): {skipped}")
    print(f"▶ Queueing tasks to run: {len(tasks)}")

    if len(tasks) == 0:
        print(
            f"\n🎉 Completed OK: {total} | Damask-only: {done_damask_only} | Skipped: {skipped + done_skipped_in_worker} | Timed out: {timed_out} | Failed: {failed}"
        )
        return

    # Progress bar (tqdm). Falls back to periodic prints if tqdm unavailable.
    try:
        from tqdm.auto import tqdm  # type: ignore

        use_tqdm = True
    except Exception:
        tqdm = None
        use_tqdm = False

    # Split into batches if local-staging is used
    if local_staging:
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        print(f"📦 Split into {len(batches)} batches of up to {batch_size}")
    else:
        batches = [tasks]

    transfer_threads = []

    # Background input prefetch: while batch N is generating, a daemon thread
    # stages batch N+1's input files onto the local SSD. This overlaps the
    # (single-reader, USB-bound) staging with the (CPU-bound) generation so the
    # workers never sit idle waiting for inputs between batches.
    prefetch_results: dict = {}
    prefetch_threads: dict = {}

    def _prefetch_batch(idx):
        if stage_inputs_dir and 0 <= idx < len(batches):
            prefetch_results[idx] = _stage_batch_inputs(batches[idx], stage_inputs_dir)

    # Kick off staging of the first batch before the loop begins.
    if stage_inputs_dir and batches:
        t0 = threading.Thread(target=_prefetch_batch, args=(0,), daemon=True)
        t0.start()
        prefetch_threads[0] = t0

    for batch_idx, batch_tasks in enumerate(batches):
        if local_staging:
            print(f"\n{'='*60}")
            print(f"🔄 Batch {batch_idx + 1}/{len(batches)} ({len(batch_tasks)} tasks)")
            print(f"{'='*60}")

        # Make sure THIS batch's inputs are fully staged (wait for its prefetch
        # thread, which usually finished during the previous batch's generation).
        staged_inputs = []
        if stage_inputs_dir:
            th = prefetch_threads.pop(batch_idx, None)
            if th is not None:
                th.join()
            staged_inputs = prefetch_results.pop(batch_idx, [])
            print(f"📥 Inputs ready for batch {batch_idx + 1}: {len(staged_inputs)} files")

        # Start prefetching the NEXT batch's inputs in the background so they are
        # ready by the time this batch finishes generating.
        if stage_inputs_dir and (batch_idx + 1) < len(batches):
            tn = threading.Thread(target=_prefetch_batch, args=(batch_idx + 1,), daemon=True)
            tn.start()
            prefetch_threads[batch_idx + 1] = tn

        # Adaptive chunksize: keep it small so tqdm updates quickly.
        chunksize = max(1, min(4, len(batch_tasks) // (num_workers * 4)))

        # Parallel execution (initializer pre-imports damask once per worker)
        pool = mp.Pool(
            processes=num_workers,
            maxtasksperchild=maxtasksperchild,
            initializer=_worker_init,
        )
        try:
            iterator = pool.imap_unordered(_process_one_sample, batch_tasks, chunksize=chunksize)
            processed = 0

            if use_tqdm:
                pbar = tqdm(
                    iterator,
                    total=len(batch_tasks),
                    desc=f"Generating RVEs [batch {batch_idx+1}/{len(batches)}]" if local_staging else "Generating RVEs",
                    unit="sample",
                    dynamic_ncols=True,
                    smoothing=0.05,
                    mininterval=0.5,
                    file=sys.stderr,
                )
            else:
                pbar = iterator

            for res in pbar:
                processed += 1
                mat = res.get("material", "?")
                sid = res.get("sample_id", "?")
                if res.get("ok"):
                    stage = res.get("stage")
                    if stage == "skipped":
                        done_skipped_in_worker += 1
                    elif stage == "done_damask_only":
                        done_damask_only += 1
                        total += 1
                    else:
                        total += 1
                else:
                    stage = res.get("stage")
                    msg = f"❌ {mat} ID={sid} failed at {stage}: {res.get('error')}"
                    if stage == "timeout":
                        timed_out += 1
                        msg = f"⏱️ {mat} ID={sid} timed out (will retry next run)"
                    else:
                        failed += 1
                    tqdm.write(msg) if use_tqdm else print(msg)

                # Update progress bar with latest sample info
                if use_tqdm:
                    try:
                        pbar.set_postfix_str(
                            f"{mat}/{sid} | ok={total} dsk={done_damask_only} skip={skipped + done_skipped_in_worker} timeout={timed_out} fail={failed}",
                            refresh=True,
                        )
                    except Exception:
                        pass

                if (not use_tqdm) and (processed % 50 == 0 or processed == len(batch_tasks)):
                    print(
                        f"✅ [{mat}/{sid}] Processed: {processed}/{len(batch_tasks)} | Done: {total} | Damask-only: {done_damask_only} | Skipped: {skipped + done_skipped_in_worker} | Timed out: {timed_out} | Failed: {failed}"
                    )
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user (Ctrl+C). Terminating workers...")
            pool.terminate()
            break
        except Exception:
            pool.terminate()
            raise
        else:
            pool.close()
        finally:
            pool.join()

        # Workers are done reading; free the local input cache for this batch.
        if staged_inputs:
            _cleanup_staged_inputs(staged_inputs)

        # After each batch: transfer to external drive in background
        # (skipped entirely in --no-transfer mode, where outputs stay on the SSD
        # and a separate encoder ships only PNGs to the USB).
        if local_staging and not no_transfer:
            # Wait for any previous transfer to finish first
            for t in transfer_threads:
                t.join()
            transfer_threads.clear()
            t = _transfer_in_background(local_staging, rve_root)
            transfer_threads.append(t)

    # Wait for final transfer to complete
    for t in transfer_threads:
        print("⏳ Waiting for final transfer to complete...")
        t.join()

    print(
        f"\n🎉 Completed OK: {total} | Damask-only: {done_damask_only} | Skipped: {skipped + done_skipped_in_worker} | Timed out (retry next run): {timed_out} | Failed: {failed}"
    )


if __name__ == "__main__":
    main()