"""
Utilities module with helper classes for statistics, file I/O, and logging.
"""

import os
import pickle
from typing import Dict, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

from .config import Config


class ClassStatistics:
    """Manages class statistics (centroids, distances, thresholds)."""
    
    def __init__(self, config: Config):
        """
        Initialize ClassStatistics.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.stats: Dict[str, dict] = {}

    @staticmethod
    def _mahalanobis_to_centroid(
        X: np.ndarray,
        centroid: np.ndarray,
        inv_cov: np.ndarray
    ) -> np.ndarray:
        """Compute Mahalanobis distances from samples to one centroid."""
        diff = X - centroid
        squared = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        squared = np.maximum(squared, 0.0)
        return np.sqrt(squared)

    def _fit_inverse_covariance(self, X: np.ndarray) -> np.ndarray:
        """Fit a stable inverse covariance matrix for sparse high-dimensional data."""
        n_features = X.shape[1]

        if len(X) < 2:
            return np.eye(n_features)

        try:
            cov = LedoitWolf().fit(X).covariance_
        except ValueError:
            cov = np.cov(X, rowvar=False)
            cov = np.atleast_2d(cov)

        cov = cov + np.eye(n_features) * 1e-6
        return np.linalg.pinv(cov)

    @staticmethod
    def _num_subclusters(n_samples: int) -> int:
        """Choose a conservative number of local class modes."""
        if n_samples < 6:
            return 1
        return min(3, max(1, n_samples // 3))
    

    def compute_statistics(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        class_names: list
    ):
        """
        Compute statistics for each class in PCA space using Mahalanobis distance.
        
        For each class:
        1. Compute global covariance using shrinkage for stability
        2. Split sufficiently populated classes into local sub-clusters
        3. Calculate each point's distance to the nearest sub-cluster centroid
        4. Compute threshold as the configured percentile of original distances
        
        Args:
            X_pca_by_class: Dict[class_name -> samples in PCA space]
            class_names: List of class names (sorted)
        """
        print("\n" + "="*80)
        print("COMPUTING CLASS STATISTICS (SUBCLUSTER MAHALANOBIS)")
        print("="*80)
        print(f"\n{'Class Name':<40} {'Modes':<8} {'Mean Dist':<12} {'Threshold':<12}")
        print("-" * 74)
        
        for class_name in class_names:
            X_class_pca = X_pca_by_class[class_name]
            n_samples = len(X_class_pca)
            
            if n_samples == 0:
                raise ValueError(f"Class {class_name} has 0 samples")
            
            centroid = X_class_pca.mean(axis=0)
            inv_cov = self._fit_inverse_covariance(X_class_pca)

            n_modes = self._num_subclusters(n_samples)
            if n_modes == 1:
                subcentroids = centroid.reshape(1, -1)
            else:
                kmeans = KMeans(
                    n_clusters=n_modes,
                    random_state=self.config.RANDOM_SEED,
                    n_init=10
                )
                kmeans.fit(X_class_pca)
                subcentroids = kmeans.cluster_centers_

            cluster_models = [
                {
                    'centroid': subcentroid,
                    'inv_cov': inv_cov,
                    'n_samples': n_samples,
                }
                for subcentroid in subcentroids
            ]

            mahal_distances = self._compute_min_mahalanobis_distance(
                X_class_pca,
                cluster_models
            )

            # Compute threshold (Xth percentile of original Mahalanobis distances)
            percentile_value = self.config.CENTROID_FILTER_PERCENTILE
            threshold = np.percentile(mahal_distances, percentile_value)
            threshold *= self.config.CENTROID_FILTER_SCALE
            
            self.stats[class_name] = {
                'centroid': centroid,
                'inv_cov': inv_cov,
                'cluster_models': cluster_models,
                'n_modes': n_modes,
                'threshold': threshold,
                'mean_dist': mahal_distances.mean(),
                'std_dist': mahal_distances.std(),
                'n_original': n_samples,
                'distances': mahal_distances
            }
            
            print(f"{class_name:<40} {n_modes:<8} "
                f"{mahal_distances.mean():>12.10f}+/-{mahal_distances.std():<11.10f} "
                f"{threshold:>17.10f}")
        
        print("="*80)
    
    def get_centroid(self, class_name: str) -> np.ndarray:
        """Get centroid for a class (for backward compatibility)."""
        return self.stats[class_name]['centroid']
    
    def get_threshold(self, class_name: str) -> float:
        """Get distance threshold for a class."""
        return self.stats[class_name]['threshold']
    
    def get_distances(self, class_name: str) -> np.ndarray:
        """Get original distances for a class."""
        return self.stats[class_name]['distances']
    

    def _compute_min_mahalanobis_distance(
        self,
        X: np.ndarray,
        cluster_models: list
    ) -> np.ndarray:
        """Compute distance to the nearest local class model."""
        if len(X) == 0:
            return np.array([])

        all_distances = []
        for model in cluster_models:
            all_distances.append(
                self._mahalanobis_to_centroid(
                    X,
                    model['centroid'],
                    model['inv_cov']
                )
            )

        return np.vstack(all_distances).min(axis=0)

    def compute_min_mahalanobis_distance(
        self, 
        X: np.ndarray, 
        class_name: str
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance for samples to the nearest class sub-cluster.
        
        Args:
            X: (n_samples, n_features) samples to compute distances for
            class_name: Class name to get centroid/covariance for
            
        Returns:
            (n_samples,) Mahalanobis distances
        """
        cluster_models = self.stats[class_name].get('cluster_models')

        if cluster_models is None:
            cluster_models = [{
                'centroid': self.stats[class_name]['centroid'],
                'inv_cov': self.stats[class_name]['inv_cov'],
                'n_samples': self.stats[class_name]['n_original'],
            }]

        return self._compute_min_mahalanobis_distance(X, cluster_models)
    
    def save(self, filepath: Optional[str] = None):
        """Save statistics to file."""
        if filepath is None:
            filepath = os.path.join(self.config.LOGS_DIR, 'class_stats.pkl')
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.stats, f)
        
        print(f"Class statistics saved to: {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """Load statistics from file."""
        if filepath is None:
            filepath = os.path.join(self.config.LOGS_DIR, 'class_stats.pkl')
        
        with open(filepath, 'rb') as f:
            self.stats = pickle.load(f)
        
        print(f"Class statistics loaded from: {filepath}")


class FileManager:
    """Manages file I/O operations for synthetic samples."""
    
    def __init__(self, config: Config):
        """
        Initialize FileManager.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def save_synthetic_samples(
        self,
        synthetic_original: Dict[str, np.ndarray],
        class_names: list
    ) -> int:
        """
        Save synthetic samples to files.
        
        Args:
            synthetic_original: Dict[class_name -> samples in original space]
            class_names: List of class names (sorted)
            
        Returns:
            Total number of files saved
        """
        print("\n" + "="*80)
        print("SAVING SYNTHETIC SAMPLES")
        print("="*80)
        
        total_saved = 0
        
        for class_name in class_names:
            samples = synthetic_original[class_name]
            
            if len(samples) == 0:
                print(f"{class_name:<40} No samples to save")
                continue
            
            # Create class directory
            class_dir = os.path.join(self.config.OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save each sample
            for idx, sample in enumerate(samples):
                # Reshape back to (9139, 2) format [real, imaginary]
                sample_reshaped = sample.reshape(self.config.ODF_SHAPE)
                
                # Round to specified decimal places
                sample_reshaped = np.round(sample_reshaped, 
                                          decimals=self.config.OUTPUT_DECIMALS)
                
                # Remove negative zeros
                sample_reshaped = np.where(sample_reshaped == 0, 0, sample_reshaped)
                
                # Generate filename
                filename = f"{class_name}_{idx+1}.txt"
                filepath = os.path.join(class_dir, filename)
                
                # Save
                np.savetxt(filepath, sample_reshaped, 
                          fmt=f'%.{self.config.OUTPUT_DECIMALS}f')
                total_saved += 1
            
            print(f"{class_name:<40} Saved {len(samples):>5} files to {class_dir}/")
        
        print("="*80)
        print(f"Total files saved: {total_saved}")
        print("="*80)
        
        return total_saved


class Logger:
    """Generates summary reports and logs."""

    def __init__(self, config: Config):
        """
        Initialize Logger.

        Args:
            config: Configuration object
        """
        self.config = config

    def generate_report(
        self,
        class_data: Dict[str, np.ndarray],
        synthetic_original: Dict[str, np.ndarray],
        class_names: list
    ):
        """
        Generate and save summary report.

        Args:
            class_data: Original data by class
            synthetic_original: Synthetic data in original space by class
            class_names: List of class names (sorted)
        """
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)

        report_path = os.path.join(self.config.LOGS_DIR, 'oversampling_report.txt')

        total_orig = sum(len(class_data[c]) for c in class_names)
        total_synth = sum(len(synthetic_original[c]) for c in class_names)
        total_all = total_orig + total_synth
        total_target = sum(
            self.config.target_synthetic_count(len(class_data[c]))
            for c in class_names
        )
        pct_total = 100 * total_synth / total_target if total_target > 0 else 0

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ODF HARMONICS OVERSAMPLING PIPELINE - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  PCA Components: {self.config.PCA_COMPONENTS}\n")
            f.write(f"  Target Synthetic Per Class: {self.config.TARGET_SYNTHETIC_PER_CLASS}\n")
            f.write(f"  Expected Per-Class Range: {self.config.MIN_SYNTHETIC_PER_CLASS}-{self.config.MAX_SYNTHETIC_PER_CLASS}\n")
            f.write(f"  Fallback Target Multiplier: {self.config.TARGET_MULTIPLIER}x\n")
            f.write(f"  Centroid Distance Threshold: {self.config.CENTROID_FILTER_PERCENTILE}th percentile\n")
            f.write(f"  Centroid Distance Scale: {self.config.CENTROID_FILTER_SCALE}x\n")
            f.write(f"  MC Latent Jitter: {self.config.MC_LATENT_JITTER}x per sample\n")
            f.write(f"  MC Latent Sigma: {self.config.MC_LATENT_SIGMA}\n")
            f.write(f"  Scale Multiplier: {self.config.SCALE_MULTIPLIER}x\n")
            f.write(f"  MixUp Multiplier: {self.config.MIXUP_MULTIPLIER}x\n")
            f.write(f"  Warp Multiplier: {self.config.WARP_MULTIPLIER}x\n")

            if self.config.USE_ADASYN:
                f.write("  Sampler: Hybrid ADASYN + Contextual Adjustment\n")
                f.write(f"  ADASYN k-neighbors: {self.config.ADASYN_K_NEIGHBORS}\n")
                f.write(f"  ADASYN Candidate Multiplier: {self.config.ADASYN_CANDIDATE_MULTIPLIER}\n")
                f.write(f"  Contextual Alpha: {self.config.CONTEXTUAL_ALPHA}\n")
                f.write(f"  Context Mode: {self.config.CONTEXT_MODE}\n")
                f.write(f"  Classifier Threshold: {self.config.CLASSIFIER_PROBA_THRESH}\n")
                f.write(f"  Geometric Margin: {self.config.GEOMETRIC_MARGIN}\n")
                f.write(f"  Filter Stage 2: {self.config.FILTER_STAGE_2}\n")
                f.write(f"  Filter Stage 3: {self.config.FILTER_STAGE_3}\n")
            else:
                f.write("  Sampler: SMOTE\n")

            f.write("\n")
            f.write("Results Per Class:\n")
            f.write(f"{'Class':<40} {'Original':<10} {'Synthetic':<10} {'Total':<10} "
                    f"{'SynthTarget':<12} {'%Target':<10}\n")
            f.write("-" * 90 + "\n")

            for class_name in class_names:
                n_orig = len(class_data[class_name])
                n_synth = len(synthetic_original[class_name])
                n_total = n_orig + n_synth
                target = self.config.target_synthetic_count(n_orig)
                pct = 100 * n_synth / target if target > 0 else 0

                f.write(f"{class_name:<40} {n_orig:<10} {n_synth:<10} {n_total:<10} "
                        f"{target:<12} {pct:<10.1f}\n")

            f.write("-" * 90 + "\n")
            f.write(f"{'TOTAL':<40} {total_orig:<10} {total_synth:<10} {total_all:<10} "
                    f"{total_target:<12} {pct_total:<10.1f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Output Directories:\n")
            f.write(f"  Synthetic Samples: {self.config.OUTPUT_DIR}/\n")
            f.write(f"  Visualizations: {self.config.VISUALIZATION_DIR}/\n")
            f.write(f"  Logs: {self.config.LOGS_DIR}/\n")

        print(f"Summary report saved to: {report_path}")

        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nTotal synthetic samples generated: {total_synth}")
        print(f"Total samples (real + synthetic): {total_all}")
        print(f"Achievement: {pct_total:.1f}% of target")
        print(f"\nCheck '{self.config.VISUALIZATION_DIR}/' for visualizations")
        print(f"Check '{self.config.OUTPUT_DIR}/' for synthetic sample files")
        print(f"Check '{self.config.LOGS_DIR}/oversampling_report.txt' for detailed report")
        print("="*80)
