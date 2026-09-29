"""
Configuration module for the ODF Oversampling Pipeline.
All hyperparameters and settings are centralized here.
"""

import math
import os
from typing import Tuple


class Config:
    """Central configuration for the ODF oversampling pipeline."""
    
    # ========================================================================
    # DIRECTORY PATHS
    # ========================================================================
    INPUT_DIR = "data/odf_harmonics"  # measured GSH coefficients, one .txt per condition
    OUTPUT_DIR = "synthetic_samples"
    VISUALIZATION_DIR = "visualizations"
    LOGS_DIR = "logs"
    
    # ========================================================================
    # PCA & PREPROCESSING
    # ========================================================================
    PCA_COMPONENTS = 110  # Number of PCA components for latent space
    
    # ========================================================================
    # TARGET GENERATION
    # ========================================================================
    TARGET_MULTIPLIER = 15  # Fallback: generate n_original * multiplier synthetic samples
    TARGET_SYNTHETIC_PER_CLASS = 7500  # Primary target: approx. 5k-10k synthetic samples per class
    MIN_SYNTHETIC_PER_CLASS = 5000  # Reporting lower bound for expected per-class yield
    MAX_SYNTHETIC_PER_CLASS = 10000  # Reporting upper bound for expected per-class yield
    
    # ========================================================================
    # AUGMENTATION PARAMETERS
    # ========================================================================
    # Monte-Carlo Latent Jittering
    MC_LATENT_JITTER = 5  # Generate 5x variants per sample
    MC_LATENT_SIGMA = 0.008  # Gaussian noise std in PCA space (conservative)
    MC_ANISOTROPIC = True  # Scale jitter by PCA component variance (True=variance-aware)
    
    # Multi-Modal Augmentation Multipliers
    SCALE_MULTIPLIER = 2  # Amplitude scaling repetitions
    MIXUP_MULTIPLIER = 1  # Local MixUp repetitions
    WARP_MULTIPLIER = 1   # Time warping repetitions
    
    # Augmentation Ranges
    SCALE_RANGE: Tuple[float, float] = (0.98, 1.02)  # Amplitude scaling range
    MIXUP_ALPHA = 0.3  # Beta distribution parameter for MixUp
    LOCAL_MIXUP_K = 5  # Restrict MixUp partners to local same-class neighbors
    MAX_WARP_FACTOR = 1.05  # Maximum time warping factor
    
    # ========================================================================
    # GENERATION MODE
    # ========================================================================
    # 'adasyn'       - k-NN edge interpolation (1D filaments between neighbours)
    # 'distribution' - subcluster-aware interior filling: Dirichlet barycentric
    #                  mixing (fills convex interior) + Gaussian sampling (fills
    #                  the ellipsoid + controlled extrapolation in the tails).
    GENERATION_MODE = 'adasyn'  # Options: 'adasyn' | 'distribution'

    # Distribution-fill parameters (used when GENERATION_MODE == 'distribution')
    BARYCENTRIC_POINTS = 4        # Points combined per Dirichlet barycentric sample (>=2)
    DIRICHLET_ALPHA = 1.0         # Dirichlet concentration (1.0 = uniform over simplex)
    GAUSSIAN_FRACTION = 0.5       # Gaussian share of the interior pool (rest = barycentric)
    GAUSSIAN_COV_SCALE = 1.15     # Covariance inflation: 1.0=interpolate, >1=extrapolate tails
    EDGE_FRACTION = 0.0           # Share of total from ADASYN/SMOTE edges (0 = no ADASYN blend)
    DISTRIBUTION_MIN_SUBCLUSTER = 6  # Min class size before splitting into subclusters
    USE_AUGMENTED_FILL_BASIS = True  # Feed the jittered+augmented halo (already computed) into
                                     # the distribution fill basis: richer covariance + more
                                     # diverse interior than the bare 2-18 real points.
    BRIDGE_FRACTION = 0.0            # Share of interior pool generated as cross-subcluster
                                    # bridges (fills inter-mode gaps; needs >=2 subclusters).
    MIN_RETENTION_FLOOR = 0         # Minimum accepted synthetic samples per class; starved
                                    # classes are rescued from the candidate pool by classifier
                                    # confidence ranking (0 = disabled).

    # ========================================================================
    # HYBRID ADASYN + CONTEXTUAL ADJUSTMENT
    # ========================================================================
    USE_ADASYN = True  # Use Hybrid ADASYN (True) or fallback to SMOTE (False)
    
    # ADASYN Parameters
    ADASYN_K_NEIGHBORS = 5  # k-neighbors for ADASYN candidate generation
    ADASYN_CANDIDATE_MULTIPLIER = 1.5  # Generate extra candidates so filters can prune
    ADASYN_OVERSAMPLE_RATIO = 1.5  # Backward-compatible alias for candidate multiplier
    
    # Contextual Adjustment Parameters
    USE_CONTEXTUAL_NUDGE = True  # Apply contextual nudge to ADASYN samples
    CONTEXTUAL_ALPHA = 0.05  # Cross-class nudge strength (smaller to reduce class leakage)
    CONTEXT_MODE = 'nearest_other_centroid'  # How to compute context vector
    # Options: 'nearest_other' | 'other_centroid' | 'nearest_other_centroid'
    
    # Filtering Thresholds
    CLASSIFIER_PROBA_THRESH = 0.90  # Minimum classifier confidence (0.8-0.9)
    GEOMETRIC_MARGIN = 1.3  # Distance ratio: d_other >= margin * d_same (1.1-1.5)
    
    # ========================================================================
    # MODULAR FILTERING STAGES
    # ========================================================================
    # Stage 1 is ALWAYS classifier confidence (mandatory)
    # Stage 2 and Stage 3 are configurable - choose from:
    #   'geometric_margin'   - Distance-based class separation filtering
    #   'centroid_distance'  - Mahalanobis distance to sub-cluster centroids
    #   'none'               - Skip this stage
    
    FILTER_STAGE_2 = 'geometric_margin'    # Options: 'geometric_margin' | 'centroid_distance' | 'none'
    FILTER_STAGE_3 = 'centroid_distance'   # Options: 'geometric_margin' | 'centroid_distance' | 'none'
    
    # ========================================================================
    # DISTANCE FILTERING
    # ========================================================================
    CENTROID_FILTER_PERCENTILE = 95  # Keep samples within original class distance envelope
    CENTROID_FILTER_SCALE = 1.0  # >1 permits controlled extrapolation beyond percentile envelope
    
    # ========================================================================
    # RANDOM FOREST CLASSIFIER (for contextual filtering)
    # ========================================================================
    RF_N_ESTIMATORS = 200
    RF_N_JOBS = -1  # Use all CPU cores
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    VIZ_DPI = 150  # Resolution for saved plots
    VIZ_FIGSIZE_LARGE = (18, 14)  # 4-panel plots
    VIZ_FIGSIZE_MEDIUM = (12, 8)  # Single plots
    VIZ_FIGSIZE_SMALL = (10, 8)  # Per-class plots
    
    # ========================================================================
    # FILE I/O
    # ========================================================================
    OUTPUT_DECIMALS = 6  # Decimal places for saved ODF files
    ODF_SHAPE = (9139, 2)  # Shape of ODF harmonics: (n_harmonics, [real, imag])
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    RANDOM_SEED = 42
    
    # ========================================================================
    # METHODS
    # ========================================================================
    
    def create_directories(self):
        """Create all output directories if they don't exist."""
        for dir_path in [self.OUTPUT_DIR, self.VISUALIZATION_DIR, self.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    def target_synthetic_count(self, n_original: int) -> int:
        """Return the desired number of final synthetic samples for one class."""
        if self.TARGET_SYNTHETIC_PER_CLASS and self.TARGET_SYNTHETIC_PER_CLASS > 0:
            return int(self.TARGET_SYNTHETIC_PER_CLASS)
        return int(n_original * self.TARGET_MULTIPLIER)

    def candidate_synthetic_count(self, n_original: int) -> int:
        """Return the number of pre-filter candidates to generate for one class."""
        target = self.target_synthetic_count(n_original)
        multiplier = max(float(self.ADASYN_CANDIDATE_MULTIPLIER), float(self.ADASYN_OVERSAMPLE_RATIO), 1.0)
        return max(target, int(math.ceil(target * multiplier)))

    def summary(self) -> str:
        """Return a formatted summary of configuration."""
        lines = [
            "="*80,
            "CONFIGURATION SUMMARY",
            "="*80,
            "",
            "Paths:",
            f"  Input Directory: {self.INPUT_DIR}",
            f"  Output Directory: {self.OUTPUT_DIR}",
            f"  Visualization Directory: {self.VISUALIZATION_DIR}",
            f"  Logs Directory: {self.LOGS_DIR}",
            "",
            "PCA & Preprocessing:",
            f"  PCA Components: {self.PCA_COMPONENTS}",
            "",
            "Target Generation:",
            f"  Target Synthetic Per Class: {self.TARGET_SYNTHETIC_PER_CLASS}",
            f"  Expected Per-Class Range: {self.MIN_SYNTHETIC_PER_CLASS}-{self.MAX_SYNTHETIC_PER_CLASS}",
            f"  Fallback Target Multiplier: {self.TARGET_MULTIPLIER}x",
            "",
            "Augmentation:",
            f"  MC Latent Jitter: {self.MC_LATENT_JITTER}x (sigma={self.MC_LATENT_SIGMA})",
            f"  Scale Multiplier: {self.SCALE_MULTIPLIER}x (range: {self.SCALE_RANGE})",
            f"  MixUp Multiplier: {self.MIXUP_MULTIPLIER}x (alpha={self.MIXUP_ALPHA}, local k={self.LOCAL_MIXUP_K})",
            f"  Warp Multiplier: {self.WARP_MULTIPLIER}x (max factor: {self.MAX_WARP_FACTOR})",
            "",
            "Generation:",
            f"  Generation Mode: {self.GENERATION_MODE}",
            f"  Barycentric Points: {self.BARYCENTRIC_POINTS} (dirichlet alpha={self.DIRICHLET_ALPHA})",
            f"  Gaussian Fraction: {self.GAUSSIAN_FRACTION} (cov scale={self.GAUSSIAN_COV_SCALE})",
            f"  ADASYN Edge Fraction: {self.EDGE_FRACTION}",
            f"  Augmented Fill Basis: {self.USE_AUGMENTED_FILL_BASIS}",
            f"  Bridge Fraction: {self.BRIDGE_FRACTION}",
            f"  Min Retention Floor: {self.MIN_RETENTION_FLOOR}",
            "",
            "Hybrid ADASYN:",
            f"  Use ADASYN: {self.USE_ADASYN}",
            f"  ADASYN k-neighbors: {self.ADASYN_K_NEIGHBORS}",
            f"  Candidate Multiplier: {self.ADASYN_CANDIDATE_MULTIPLIER}x",
            f"  Contextual Alpha: {self.CONTEXTUAL_ALPHA}",
            f"  Context Mode: {self.CONTEXT_MODE}",
            f"  Classifier Threshold: {self.CLASSIFIER_PROBA_THRESH}",
            f"  Geometric Margin: {self.GEOMETRIC_MARGIN}",
            "",
            "Filtering:",
            f"  Stage 2: {self.FILTER_STAGE_2}",
            f"  Stage 3: {self.FILTER_STAGE_3}",
            f"  Centroid Distance Percentile: {self.CENTROID_FILTER_PERCENTILE}%",
            f"  Centroid Distance Scale: {self.CENTROID_FILTER_SCALE}x",
            "",
            "Reproducibility:",
            f"  Random Seed: {self.RANDOM_SEED}",
            "="*80
        ]
        return "\n".join(lines)
    
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Check PCA components
        if self.PCA_COMPONENTS <= 0:
            errors.append("PCA_COMPONENTS must be positive")
        
        # Check multipliers
        if self.TARGET_MULTIPLIER <= 0:
            errors.append("TARGET_MULTIPLIER must be positive")
        if self.TARGET_SYNTHETIC_PER_CLASS and self.TARGET_SYNTHETIC_PER_CLASS > 0:
            if self.MIN_SYNTHETIC_PER_CLASS > self.TARGET_SYNTHETIC_PER_CLASS:
                errors.append("MIN_SYNTHETIC_PER_CLASS must be <= TARGET_SYNTHETIC_PER_CLASS")
            if self.TARGET_SYNTHETIC_PER_CLASS > self.MAX_SYNTHETIC_PER_CLASS:
                errors.append("TARGET_SYNTHETIC_PER_CLASS must be <= MAX_SYNTHETIC_PER_CLASS")
        if self.MC_LATENT_JITTER <= 0:
            errors.append("MC_LATENT_JITTER must be positive")
        
        # Check ranges
        if self.SCALE_RANGE[0] >= self.SCALE_RANGE[1]:
            errors.append("SCALE_RANGE must be (min, max) with min < max")
        if not (0 < self.MIXUP_ALPHA <= 1):
            errors.append("MIXUP_ALPHA must be in (0, 1]")
        if self.LOCAL_MIXUP_K <= 0:
            errors.append("LOCAL_MIXUP_K must be positive")
        if self.MAX_WARP_FACTOR <= 1:
            errors.append("MAX_WARP_FACTOR must be > 1")
        
        # Check generation mode
        if self.GENERATION_MODE not in ['adasyn', 'distribution']:
            errors.append("GENERATION_MODE must be one of: 'adasyn', 'distribution'")
        if self.BARYCENTRIC_POINTS < 2:
            errors.append("BARYCENTRIC_POINTS must be >= 2")
        if self.DIRICHLET_ALPHA <= 0:
            errors.append("DIRICHLET_ALPHA must be positive")
        if not (0 <= self.GAUSSIAN_FRACTION <= 1):
            errors.append("GAUSSIAN_FRACTION must be in [0, 1]")
        if not (0 <= self.EDGE_FRACTION <= 1):
            errors.append("EDGE_FRACTION must be in [0, 1]")
        if not (0 <= self.BRIDGE_FRACTION <= 1):
            errors.append("BRIDGE_FRACTION must be in [0, 1]")
        if self.MIN_RETENTION_FLOOR < 0:
            errors.append("MIN_RETENTION_FLOOR must be >= 0")
        if self.GAUSSIAN_COV_SCALE <= 0:
            errors.append("GAUSSIAN_COV_SCALE must be positive")

        # Check ADASYN parameters
        if self.ADASYN_K_NEIGHBORS <= 0:
            errors.append("ADASYN_K_NEIGHBORS must be positive")
        if self.ADASYN_CANDIDATE_MULTIPLIER < 1:
            errors.append("ADASYN_CANDIDATE_MULTIPLIER must be >= 1")
        if not (0 < self.CONTEXTUAL_ALPHA < 1):
            errors.append("CONTEXTUAL_ALPHA must be in (0, 1)")
        if self.CONTEXT_MODE not in ['nearest_other', 'other_centroid', 'nearest_other_centroid']:
            errors.append("CONTEXT_MODE must be one of: 'nearest_other', 'other_centroid', 'nearest_other_centroid'")
        
        # Check filtering thresholds
        if not (0 < self.CLASSIFIER_PROBA_THRESH <= 1):
            errors.append("CLASSIFIER_PROBA_THRESH must be in (0, 1]")
        if self.GEOMETRIC_MARGIN <= 1:
            errors.append("GEOMETRIC_MARGIN must be > 1")
        if not (0 < self.CENTROID_FILTER_PERCENTILE <= 100):
            errors.append("CENTROID_FILTER_PERCENTILE must be in (0, 100]")
        if self.CENTROID_FILTER_SCALE <= 0:
            errors.append("CENTROID_FILTER_SCALE must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
