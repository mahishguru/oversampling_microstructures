"""
Augmentation module for generating synthetic variants.
Includes Monte-Carlo jittering, amplitude scaling, mixup, and time warping.
"""

from typing import Dict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

from .config import Config


class Augmentor:
    """Applies various augmentation techniques in PCA latent space."""
    
    def __init__(self, config: Config):
        """
        Initialize Augmentor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        np.random.seed(config.RANDOM_SEED)
    
    def monte_carlo_jitter(self, X_pca: np.ndarray, anisotropic: bool = True) -> np.ndarray:
        """
        Apply Monte Carlo jittering in PCA space with variance-aware perturbations.
        
        For each sample x_i, generates M jittered samples:
            x̃_i,m = x_i + ε_i,m
        where ε ~ N(0, Σ) and Σ is either:
        - Anisotropic: Σ = diag(σ² * var_per_dim) — scales by PCA component variance
        - Isotropic: Σ = σ²I — uniform noise across all dimensions
        
        Mathematical foundation:
            𝔼[ℓ(f(x + ε), y)] ≈ (1/M) Σ_m ℓ(f(x + ε_m), y)
        
        This approximates the expectation over perturbations using Monte Carlo sampling,
        encouraging local smoothness: 𝔼[g(x+ε)] ≈ g(x) + ½σ² Tr(H_g(x))
        
        Args:
            X_pca: (n_samples, n_components) array in PCA space
            anisotropic: If True, scale noise by empirical variance per PCA dimension
            
        Returns:
            X_jittered: (n_samples * MC_LATENT_JITTER, n_components) jittered samples
        """
        n_samples, n_components = X_pca.shape
        M = self.config.MC_LATENT_JITTER  # Number of jitters per sample
        sigma = self.config.MC_LATENT_SIGMA
        
        if anisotropic:
            # Compute empirical variance per PCA component
            # Earlier PCs have higher variance, should get proportionally larger jitter
            var_per_dim = np.var(X_pca, axis=0, ddof=1)  # shape: (n_components,)
            std_per_dim = np.sqrt(var_per_dim)
            
            # Scale noise by (sigma * std_per_dim)
            # This makes jitter proportional to data spread in each PC direction
            Z = np.random.standard_normal(size=(n_samples * M, n_components))
            eps = Z * (sigma * std_per_dim)[None, :]  # Broadcast: (n*M, d) * (1, d)
        else:
            # Isotropic: uniform noise across all dimensions
            eps = np.random.normal(
                loc=0.0, 
                scale=sigma, 
                size=(n_samples * M, n_components)
            )
        
        # Repeat each sample M times: [x1, x1, ..., x2, x2, ..., xn, xn]
        X_repeated = np.repeat(X_pca, M, axis=0)  # shape: (n_samples * M, n_components)
        
        # Apply Monte Carlo jittering: x̃ = x + ε
        X_jittered = X_repeated + eps
        
        return X_jittered
    
    def amplitude_scaling(self, z: np.ndarray) -> np.ndarray:
        """
        Apply energy-preserving amplitude scaling in PCA space.
        
        Scales the L2 norm (signal energy) while preserving direction:
            z_scaled = (z / ||z||) * (||z|| * scale)
        
        This is physically meaningful for microstructure descriptors, as it
        modulates the overall "intensity" of the ODF pattern while keeping
        the directional structure intact.
        
        Args:
            z: (n_components,) array in PCA space
            
        Returns:
            z_scaled: Energy-scaled version with same direction, scaled magnitude
        """
        scale = np.random.uniform(
            self.config.SCALE_RANGE[0], 
            self.config.SCALE_RANGE[1]
        )
        
        # Energy-preserving scaling: scale L2 norm while keeping direction
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-10:  # Avoid division by zero for near-zero vectors
            return z * scale
        
        z_scaled = (z / z_norm) * (z_norm * scale)
        return z_scaled
    
    def mixup(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """
        MixUp interpolation between two samples.
        
        Args:
            z1: (n_components,) first sample in PCA space
            z2: (n_components,) second sample in PCA space
            
        Returns:
            z_mixed: Interpolated sample
        """
        lam = np.random.beta(self.config.MIXUP_ALPHA, self.config.MIXUP_ALPHA)
        return lam * z1 + (1 - lam) * z2
    
    def time_warp(self, z: np.ndarray) -> np.ndarray:
        """
        Apply smooth local amplitude scaling (time warping) in PCA space.
        
        Generates a smooth, spatially-varying scaling profile using Gaussian-filtered
        random noise, then applies it component-wise. This creates gradual variations
        in component amplitudes rather than sharp segmented changes.
        
        Mathematical formulation:
            noise ~ N(0, α), α = MAX_WARP_FACTOR - 1
            s = 1 + GaussianFilter(noise, σ=2.0)
            s_normalized = s / mean(s)  [keeps mean scaling = 1]
            z_warped = z * s_normalized
        
        Physical interpretation:
            - Simulates smooth variations in PCA component contributions
            - Preserves overall signal structure (mean scaling = 1)
            - Gaussian smoothing ensures spatial continuity across components
        
        Args:
            z: (n_components,) array in PCA space
            
        Returns:
            z_warped: Smoothly time-warped version with local amplitude variations
        """
        n_components = len(z)
        
        # Maximum relative scaling strength
        # If MAX_WARP_FACTOR = 1.05, then alpha = 0.05 (±5% variation)
        alpha = self.config.MAX_WARP_FACTOR - 1.0
        
        # Generate random small deviations around 0
        noise = np.random.normal(0, alpha, size=n_components)
        
        # Apply Gaussian smoothing for spatial continuity
        # sigma=2.0 provides smooth variation across ~5 components
        smoothed_noise = gaussian_filter1d(noise, sigma=2.0)
        
        # Scaling profile: baseline 1.0 + smooth deviations
        s = 1.0 + smoothed_noise
        
        # Normalize to keep mean scaling = 1 (preserves overall energy)
        s /= np.mean(s)
        
        # Apply smooth local scaling
        z_warped = z * s
        
        return z_warped
    
    def apply_multi_modal_augmentation(
        self, 
        X_jittered_pca: np.ndarray
    ) -> np.ndarray:
        """
        Apply all augmentation techniques to jittered samples.
        
        For each jittered sample:
        - Apply amplitude scaling SCALE_MULTIPLIER times
        - Apply mixup MIXUP_MULTIPLIER times
        - Apply time warping WARP_MULTIPLIER times
        
        Args:
            X_jittered_pca: (n_jittered, n_components) jittered samples in PCA space
            
        Returns:
            X_augmented: (n_augmented, n_components) augmented samples
        """
        n_jittered = len(X_jittered_pca)
        aug_list = []

        if n_jittered > 1 and self.config.MIXUP_MULTIPLIER > 0:
            # kneighbors(return_distance=False) excludes each point itself, so
            # n_neighbors must be strictly less than the number of samples.
            n_neighbors = min(self.config.LOCAL_MIXUP_K + 1, n_jittered - 1)
            local_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(X_jittered_pca)
            neighbor_indices = local_neighbors.kneighbors(return_distance=False)
        else:
            neighbor_indices = None
        
        for i in range(n_jittered):
            z = X_jittered_pca[i]
            
            # 1. Amplitude scaling
            for _ in range(self.config.SCALE_MULTIPLIER):
                aug_list.append(self.amplitude_scaling(z))
            
            # 2. Local MixUp with a nearby jittered sample from the same class
            if neighbor_indices is not None:
                local_candidates = neighbor_indices[i]
                local_candidates = local_candidates[local_candidates != i]
                if len(local_candidates) > 0:
                    for _ in range(self.config.MIXUP_MULTIPLIER):
                        idx_other = np.random.choice(local_candidates)
                        z_other = X_jittered_pca[idx_other]
                        aug_list.append(self.mixup(z, z_other))
            
            # 3. Local time warping
            for _ in range(self.config.WARP_MULTIPLIER):
                aug_list.append(self.time_warp(z))
        
        if len(aug_list) == 0:
            return np.empty((0, X_jittered_pca.shape[1]))

        return np.array(aug_list)
    
    def augment_class_data(
        self, 
        X_pca_by_class: Dict[str, np.ndarray], 
        class_names: list
    ) -> tuple:
        """
        Apply full augmentation pipeline to all classes.
        
        Args:
            X_pca_by_class: Dict mapping class names to PCA-transformed data
            class_names: List of class names (sorted)
            
        Returns:
            Tuple of (jittered_pca, augmented_pca) where:
                jittered_pca: Dict[class_name -> jittered samples]
                augmented_pca: Dict[class_name -> augmented samples]
        """
        print("\n" + "="*80)
        print("AUGMENTATION PIPELINE")
        print("="*80)
        
        # Step 1: Monte-Carlo jittering
        jitter_mode = "Anisotropic (variance-aware)" if self.config.MC_ANISOTROPIC else "Isotropic"
        print(f"\n[1/2] Monte-Carlo Jittering ({self.config.MC_LATENT_JITTER}x per sample, {jitter_mode})")
        print("-" * 80)
        jittered_pca = {}
        
        for class_name in class_names:
            X_class_pca = X_pca_by_class[class_name]
            X_jittered = self.monte_carlo_jitter(X_class_pca, anisotropic=self.config.MC_ANISOTROPIC)
            jittered_pca[class_name] = X_jittered
            
            print(f"{class_name:<40} {len(X_class_pca):>5} → {len(X_jittered):>5} samples")
        
        # Step 2: Multi-modal augmentation
        print(f"\n[2/2] Multi-Modal Augmentation (Scale:{self.config.SCALE_MULTIPLIER}x + "
              f"MixUp:{self.config.MIXUP_MULTIPLIER}x + Warp:{self.config.WARP_MULTIPLIER}x)")
        print("-" * 80)
        augmented_pca = {}
        total_multiplier = (
            self.config.SCALE_MULTIPLIER + 
            self.config.MIXUP_MULTIPLIER + 
            self.config.WARP_MULTIPLIER
        )
        
        for class_name in class_names:
            X_jittered = jittered_pca[class_name]
            X_augmented = self.apply_multi_modal_augmentation(X_jittered)
            augmented_pca[class_name] = X_augmented
            
            print(f"{class_name:<40} {len(X_jittered):>5} → {len(X_augmented):>5} samples")
        
        print("\n" + "="*80)
        print(f"Augmentation complete: Each original sample → "
              f"{self.config.MC_LATENT_JITTER * total_multiplier}x augmented samples")
        print("="*80)
        
        return jittered_pca, augmented_pca
