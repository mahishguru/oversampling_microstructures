#!/usr/bin/env python3
"""
Main pipeline orchestrator for ODF Harmonics Oversampling.

This script coordinates all modules to execute the complete pipeline:
1. Data loading and organization
2. PCA transformation
3. Class statistics computation
4. Augmentation (MC jittering + multi-modal)
5. Hybrid ADASYN + contextual adjustment
6. Distance-based filtering
7. Inverse PCA transformation
8. Visualization generation
9. File saving and reporting
"""

import sys
import argparse
import os
import warnings
import numpy as np

# Import all pipeline modules
from odf_pipeline.config import Config
from odf_pipeline.data_loader import DataLoader
from odf_pipeline.pca_transform import PCATransformer
from odf_pipeline.augmentation import Augmentor
from odf_pipeline.sampling import HybridSampler
from odf_pipeline.filtering import QualityFilter
from odf_pipeline.visualization import Visualizer
from odf_pipeline.utils import ClassStatistics, FileManager, Logger

# Suppress warnings
warnings.filterwarnings('ignore')


def build_config(argv=None):
    """Build configuration and optionally redirect outputs to an experiment folder."""
    parser = argparse.ArgumentParser(description="ODF harmonics oversampling pipeline")
    parser.add_argument(
        "--profile",
        choices=["strict", "generative", "interpolation"],
        default="strict",
        help="strict/generative use ADASYN edge interpolation; interpolation uses "
             "distribution-aware interior filling (barycentric + Gaussian)."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Folder of measured GSH coefficient files (default: data/odf_harmonics)."
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Synthetic samples per class (paper: 7500). Smaller values give quick test runs; "
             "the retention floor is capped at 10%% of this value."
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Save outputs under experiments/<name>/ instead of the default output folders."
    )
    args = parser.parse_args(argv)

    config = Config()

    if args.profile == "generative":
        config.MC_LATENT_JITTER = 8
        config.MC_LATENT_SIGMA = 0.012
        config.SCALE_MULTIPLIER = 3
        config.MIXUP_MULTIPLIER = 2
        config.WARP_MULTIPLIER = 2
        config.SCALE_RANGE = (0.96, 1.04)
        config.LOCAL_MIXUP_K = 8
        config.ADASYN_CANDIDATE_MULTIPLIER = 3.0
        config.ADASYN_OVERSAMPLE_RATIO = 3.0
        config.CONTEXTUAL_ALPHA = 0.03
        config.CONTEXT_MODE = 'nearest_other_centroid'
        config.CLASSIFIER_PROBA_THRESH = 0.78
        config.GEOMETRIC_MARGIN = 1.08
        config.CENTROID_FILTER_PERCENTILE = 99
        config.CENTROID_FILTER_SCALE = 1.20

    elif args.profile == "interpolation":
        # Distribution-aware interior filling: less local perturbation, more
        # class- and global-distribution-aware interpolation + bounded extrapolation.
        config.GENERATION_MODE = 'distribution'
        # Keep augmentation light (only used as classifier context, not as the basis)
        config.MC_LATENT_JITTER = 3
        config.MC_LATENT_SIGMA = 0.010
        config.SCALE_MULTIPLIER = 1
        config.MIXUP_MULTIPLIER = 1
        config.WARP_MULTIPLIER = 1
        # Interior-fill controls
        config.BARYCENTRIC_POINTS = 4
        config.DIRICHLET_ALPHA = 1.0
        config.GAUSSIAN_FRACTION = 0.5
        config.GAUSSIAN_COV_SCALE = 1.25   # bounded extrapolation into the tails
        config.EDGE_FRACTION = 0.30        # keep ADASYN boundary-aware edges in the blend
        config.USE_AUGMENTED_FILL_BASIS = True  # richer covariance + more diverse interior
        config.BRIDGE_FRACTION = 0.25      # fill inter-mode gaps (global continuity)
        config.MIN_RETENTION_FLOOR = 750   # never leave sparse classes empty
        config.ADASYN_CANDIDATE_MULTIPLIER = 2.0
        config.ADASYN_OVERSAMPLE_RATIO = 2.0
        # Class-aware nudge + permissive filters so the lively interior survives
        config.USE_CONTEXTUAL_NUDGE = True
        config.CONTEXTUAL_ALPHA = 0.03
        config.CONTEXT_MODE = 'nearest_other_centroid'
        config.CLASSIFIER_PROBA_THRESH = 0.60
        config.GEOMETRIC_MARGIN = 1.05
        config.CENTROID_FILTER_PERCENTILE = 99
        config.CENTROID_FILTER_SCALE = 1.30

    if args.input_dir:
        config.INPUT_DIR = args.input_dir

    if args.target_per_class:
        config.TARGET_SYNTHETIC_PER_CLASS = int(args.target_per_class)
        config.MIN_RETENTION_FLOOR = min(config.MIN_RETENTION_FLOOR, int(args.target_per_class) // 10)
        config.MIN_SYNTHETIC_PER_CLASS = min(config.MIN_SYNTHETIC_PER_CLASS, int(args.target_per_class))

    if args.experiment_name:
        experiment_root = os.path.join("experiments", args.experiment_name)
        config.OUTPUT_DIR = os.path.join(experiment_root, "synthetic_samples")
        config.VISUALIZATION_DIR = os.path.join(experiment_root, "visualizations")
        config.LOGS_DIR = os.path.join(experiment_root, "logs")

    return config


def main(argv=None):
    """Execute the complete ODF oversampling pipeline."""
    
    print("="*80)
    print("ODF HARMONICS HYBRID OVERSAMPLING PIPELINE v2.0")
    print("="*80)
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    # Create config and validate
    config = build_config(argv)
    try:
        config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1
    
    # Set random seed
    np.random.seed(config.RANDOM_SEED)
    
    # Create output directories
    config.create_directories()
    
    # Print configuration
    print(config.summary())
    
    # Initialize all modules
    data_loader = DataLoader(config)
    pca_transformer = PCATransformer(config)
    augmentor = Augmentor(config)
    sampler = HybridSampler(config)
    quality_filter = QualityFilter(config)
    visualizer = Visualizer(config)
    class_stats = ClassStatistics(config)
    file_manager = FileManager(config)
    logger = Logger(config)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    try:
        X_all, y_all = data_loader.load_data()
        data_loader.print_summary()
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        return 1
    
    class_names = data_loader.class_names_sorted
    class_data = data_loader.class_data
    
    # ========================================================================
    # STEP 2: FIT PCA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: PCA TRANSFORMATION")
    print("="*80)
    
    X_pca = pca_transformer.fit(X_all)
    pca_transformer.save()
    
    # Split by class
    X_pca_by_class = {}
    idx = 0
    for class_name in class_names:
        n_samples = len(class_data[class_name])
        X_pca_by_class[class_name] = X_pca[idx:idx+n_samples]
        idx += n_samples
    
    # ========================================================================
    # STEP 3: COMPUTE CLASS STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: CLASS STATISTICS")
    print("="*80)
    
    class_stats.compute_statistics(X_pca_by_class, class_names)
    class_stats.save()
    
    # ========================================================================
    # STEP 4: VISUALIZE ORIGINAL DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: VISUALIZING ORIGINAL DATA")
    print("="*80)
    
    visualizer.plot_original_data(X_pca_by_class, class_names)
    
    # ========================================================================
    # STEP 5: AUGMENTATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: DATA AUGMENTATION")
    print("="*80)
    
    jittered_pca, augmented_pca = augmentor.augment_class_data(
        X_pca_by_class, class_names
    )
    
    # ========================================================================
    # STEP 6: HYBRID ADASYN SAMPLING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: ADASYN-BASED SYNTHETIC SAMPLE GENERATION")
    print("="*80)
    
    synthetic_generated_pca = sampler.augment_classes(
        X_pca_by_class,
        jittered_pca,
        augmented_pca,
        class_names
    )
    
    # ========================================================================
    # STEP 7: THREE-STAGE QUALITY FILTERING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: THREE-STAGE QUALITY FILTERING")
    print("="*80)
    
    # Prepare pooled data for classifier training
    all_samples = []
    all_labels = []
    
    for i, class_name in enumerate(class_names):
        X_combined = np.vstack([
            X_pca_by_class[class_name],
            jittered_pca[class_name],
            augmented_pca[class_name]
        ])
        all_samples.append(X_combined)
        all_labels.extend([i] * len(X_combined))
    
    X_pooled = np.vstack(all_samples)
    y_pooled = np.array(all_labels)
    
    # Train classifier for Stage 1 filtering
    quality_filter.train_classifier(X_pooled, y_pooled)
    
    # Apply all three filtering stages
    filtered_pca = quality_filter.apply_all_filters(
        synthetic_generated_pca,
        X_pooled,
        y_pooled,
        class_stats,
        class_names
    )

    filtered_pca = quality_filter.cap_to_target_counts(
        filtered_pca,
        class_stats,
        class_names
    )

    # Rescue starved classes (too sparse to survive strict filters) from the
    # pre-filter candidate pool, ranked by classifier confidence.
    filtered_pca = quality_filter.apply_retention_floor(
        filtered_pca,
        synthetic_generated_pca,
        class_names
    )
    
    # ========================================================================
    # STEP 8: INVERSE PCA TRANSFORMATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: INVERSE PCA TRANSFORMATION")
    print("="*80)
    
    synthetic_original = {}
    for class_name in class_names:
        X_synth_pca = filtered_pca[class_name]
        
        if len(X_synth_pca) == 0:
            print(f"{class_name:<40} No synthetic samples")
            synthetic_original[class_name] = np.empty((0, 18278))
            continue
        
        # Transform back to original space
        X_synth_original = pca_transformer.inverse_transform(X_synth_pca)
        synthetic_original[class_name] = X_synth_original
        
        n_orig = len(class_data[class_name])
        n_synth = len(X_synth_original)
        target = config.target_synthetic_count(n_orig)
        achievement_pct = (n_synth / target) * 100 if target > 0 else 0
        
        print(f"{class_name:<40} Original: {n_orig:<4} | Synthetic: {n_synth:<5} | "
              f"Target: {target:<5} | Achievement: {achievement_pct:.1f}%")
    
    # ========================================================================
    # STEP 9: GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualizer.generate_all_visualizations(
        X_pca_by_class,
        filtered_pca,
        class_stats,
        class_names
    )
    
    # ========================================================================
    # STEP 10: SAVE SYNTHETIC SAMPLES
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 10: SAVING SYNTHETIC SAMPLES")
    print("="*80)
    
    total_saved = file_manager.save_synthetic_samples(
        synthetic_original,
        class_names
    )
    
    # ========================================================================
    # STEP 11: GENERATE REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 11: GENERATING SUMMARY REPORT")
    print("="*80)
    
    logger.generate_report(
        class_data,
        synthetic_original,
        class_names
    )
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
