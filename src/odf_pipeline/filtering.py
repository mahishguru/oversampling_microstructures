"""
Filtering module for quality control with three-stage filtering.

Implements three complementary filtering strategies:
1. Classifier confidence filtering - ML-based quality check
2. Geometric margin filtering - Distance-based class separation
3. Distance filtering - Alignment with original data distribution
"""

from typing import Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from .config import Config
from .utils import ClassStatistics


class QualityFilter:
    """
    Three-stage quality filter for synthetic samples.
    
    Stage 1: Classifier Confidence - Ensures samples are confidently classified
    Stage 2: Geometric Margin - Ensures samples are well-separated from other classes
    Stage 3: Distance - Ensures samples align with original data distribution
    """
    
    def __init__(self, config: Config):
        """
        Initialize QualityFilter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.classifier: RandomForestClassifier = None
        self.class_mapping: Dict[int, int] = {}
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray):
        """
        Train Random Forest classifier for confidence filtering.
        
        Args:
            X: (n_samples, n_components) data in PCA space
            y: (n_samples,) class labels
        """
        print(f"Training classifier for filtering ({self.config.RF_N_ESTIMATORS} trees)...")
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            random_state=self.config.RANDOM_SEED,
            n_jobs=self.config.RF_N_JOBS
        )
        self.classifier.fit(X, y)
        
        # Cache class mapping for faster lookup
        self.class_mapping = {cls: idx for idx, cls in enumerate(self.classifier.classes_)}
        
        print(f"  ✓ Classifier trained ({len(self.class_mapping)} classes)")
    
    def filter_by_classifier_confidence(
        self,
        synthetic_pca: Dict[str, np.ndarray],
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Filter synthetic samples by classifier confidence.
        
        Stage 1: Keeps only samples where the classifier is confident (≥ threshold)
        about their class assignment.
        
        Args:
            synthetic_pca: Dict[class_name -> synthetic samples in PCA space]
            class_names: List of class names (sorted)
            
        Returns:
            filtered_pca: Dict[class_name -> filtered synthetic samples]
        """
        print("\n" + "="*80)
        print("STAGE 1: CLASSIFIER CONFIDENCE FILTERING")
        print("="*80)
        print(f"Threshold: {self.config.CLASSIFIER_PROBA_THRESH:.2f}")
        print("-" * 80)
        
        filtered_pca = {}
        
        for i, class_name in enumerate(class_names):
            X_synthetic = synthetic_pca[class_name]
            
            if len(X_synthetic) == 0:
                filtered_pca[class_name] = X_synthetic
                print(f"{class_name:<40} No samples to filter")
                continue
            
            # Get classifier predictions
            proba = self.classifier.predict_proba(X_synthetic)
            cls_idx = self.class_mapping.get(i, i)
            
            # Keep samples with high confidence
            confidence = proba[:, cls_idx]
            keep_mask = confidence >= self.config.CLASSIFIER_PROBA_THRESH
            
            X_filtered = X_synthetic[keep_mask]
            filtered_pca[class_name] = X_filtered
            
            kept_pct = 100 * len(X_filtered) / len(X_synthetic)
            print(f"{class_name:<40} {len(X_filtered):>5} / {len(X_synthetic):>5} "
                  f"({kept_pct:>5.1f}% kept)")
        
        print("="*80)
        
        return filtered_pca
    
    def filter_by_geometric_margin(
        self,
        synthetic_pca: Dict[str, np.ndarray],
        pooled_data: np.ndarray,
        pooled_labels: np.ndarray,
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Filter synthetic samples by geometric margin (class separation).
        
        Stage 2: Keeps only samples that are well-separated from other classes.
        For each sample, checks: d_other >= margin * d_same
        where d_same is distance to nearest same-class sample,
        and d_other is distance to nearest other-class sample.
        
        Args:
            synthetic_pca: Dict[class_name -> synthetic samples in PCA space]
            pooled_data: All pooled data (for neighbor search)
            pooled_labels: Labels for pooled data
            class_names: List of class names (sorted)
            
        Returns:
            filtered_pca: Dict[class_name -> filtered synthetic samples]
        """
        print("\n" + "="*80)
        print("STAGE 2: GEOMETRIC MARGIN FILTERING")
        print("="*80)
        print(f"Margin: {self.config.GEOMETRIC_MARGIN:.2f} (d_other >= {self.config.GEOMETRIC_MARGIN} x d_same)")
        print("-" * 80)
        
        filtered_pca = {}
        
        for i, class_name in enumerate(class_names):
            X_synthetic = synthetic_pca[class_name]
            
            if len(X_synthetic) == 0:
                filtered_pca[class_name] = X_synthetic
                print(f"{class_name:<40} No samples to filter")
                continue
            
            # Get same-class and other-class data
            Xc = pooled_data[pooled_labels == i]
            X_other = pooled_data[pooled_labels != i]
            
            if len(Xc) < 2:
                # Not enough same-class samples for comparison
                filtered_pca[class_name] = X_synthetic
                print(f"{class_name:<40} Skipping (insufficient samples)")
                continue
            
            # Setup neighbor searches
            nbrs_same = NearestNeighbors(n_neighbors=1).fit(Xc)

            # Batched nearest-neighbour distances (one call instead of one per sample).
            d_same = nbrs_same.kneighbors(X_synthetic, return_distance=True)[0][:, 0]
            if len(X_other) > 0:
                nbrs_other = NearestNeighbors(n_neighbors=1).fit(X_other)
                d_other = nbrs_other.kneighbors(X_synthetic, return_distance=True)[0][:, 0]
            else:
                d_other = np.full(len(X_synthetic), np.inf)

            keep_mask = d_other >= self.config.GEOMETRIC_MARGIN * (d_same + 1e-9)
            X_filtered = X_synthetic[keep_mask]
            filtered_pca[class_name] = X_filtered
            
            kept_pct = 100 * len(X_filtered) / len(X_synthetic)
            print(f"{class_name:<40} {len(X_filtered):>5} / {len(X_synthetic):>5} "
                  f"({kept_pct:>5.1f}% kept)")
        
        print("="*80)
        
        return filtered_pca

    def cap_to_target_counts(
        self,
        synthetic_pca: Dict[str, np.ndarray],
        class_stats: ClassStatistics,
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Cap each class to the configured final synthetic target.

        If more samples survive filtering than requested, keep the candidates with
        the smallest subcluster Mahalanobis distances. If too few survive, keep all
        survivors and print a warning so the candidate multiplier can be increased.
        """
        print("\n" + "="*80)
        print("TARGET COUNT CAPPING")
        print("="*80)
        print(f"Expected per-class range: {self.config.MIN_SYNTHETIC_PER_CLASS}-{self.config.MAX_SYNTHETIC_PER_CLASS}")
        print("-" * 80)

        capped_pca = {}

        for class_name in class_names:
            X_synthetic = synthetic_pca[class_name]
            n_original = class_stats.stats[class_name]['n_original']
            target = self.config.target_synthetic_count(n_original)

            if len(X_synthetic) <= target:
                capped_pca[class_name] = X_synthetic
                status = "OK"
                if len(X_synthetic) < self.config.MIN_SYNTHETIC_PER_CLASS:
                    status = "LOW"
                print(f"{class_name:<40} {len(X_synthetic):>5} / {target:>5} kept ({status})")
                continue

            distances = class_stats.compute_min_mahalanobis_distance(
                X_synthetic,
                class_name
            )
            keep_indices = np.argsort(distances, kind='mergesort')[:target]
            capped_pca[class_name] = X_synthetic[keep_indices]
            print(f"{class_name:<40} {len(X_synthetic):>5} -> {target:>5} capped")

        print("="*80)
        return capped_pca
    
    def apply_retention_floor(
        self,
        filtered_pca: Dict[str, np.ndarray],
        candidate_pca: Dict[str, np.ndarray],
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Rescue starved classes that fell below MIN_RETENTION_FLOOR.

        For any class with fewer accepted samples than the floor, pull the most
        class-consistent candidates from the (pre-filter) candidate pool, ranked by
        the already-trained classifier's confidence. This guarantees non-empty,
        plausible output for classes too sparse to survive the strict filters
        (e.g. 2-original classes) by keeping the *best* candidates rather than
        injecting indiscriminate noise. Disabled when MIN_RETENTION_FLOOR == 0.
        """
        floor = self.config.MIN_RETENTION_FLOOR
        if floor <= 0:
            return filtered_pca

        print("\n" + "="*80)
        print("RETENTION FLOOR (STARVED-CLASS RESCUE)")
        print("="*80)
        print(f"Floor: {floor} samples/class (rank by classifier confidence)")
        print("-" * 80)

        rescued = dict(filtered_pca)
        for i, class_name in enumerate(class_names):
            n_have = len(filtered_pca[class_name])
            if n_have >= floor:
                continue

            candidates = candidate_pca[class_name]
            if len(candidates) == 0:
                print(f"{class_name:<40} {n_have:>5} (no candidates to rescue)")
                continue

            cls_idx = self.class_mapping.get(i, i)
            proba = self.classifier.predict_proba(candidates)[:, cls_idx]
            n_take = min(floor, len(candidates))
            keep = np.argsort(-proba, kind='mergesort')[:n_take]
            rescued[class_name] = candidates[keep]
            print(f"{class_name:<40} {n_have:>5} -> {len(keep):>5} rescued "
                  f"(mean conf {proba[keep].mean():.2f})")

        print("="*80)
        return rescued
    
    def filter_by_centroid_distance(
        self,
        synthetic_pca: Dict[str, np.ndarray],
        class_stats: ClassStatistics,
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Filter synthetic samples by Mahalanobis distance to sub-cluster centroids.
        
        Stage 3: Keeps only samples within the threshold Mahalanobis distance from the
        nearest sub-cluster centroid (95th percentile of original samples' minimum distances).
        This ensures synthetic samples align with the original data distribution and
        handles multi-modal class distributions.
        
        Uses k-means clustering to find sub-clusters within each class, then computes
        Mahalanobis distance respecting the covariance structure of each sub-cluster.
        
        Args:
            synthetic_pca: Dict[class_name -> synthetic samples in PCA space]
            class_stats: ClassStatistics object with sub-cluster info (from original data)
            class_names: List of class names (sorted)
            
        Returns:
            filtered_pca: Dict[class_name -> filtered synthetic samples]
        """
        print("\n" + "="*80)
        print("STAGE 3: MAHALANOBIS DISTANCE FILTERING (MULTI-MODAL AWARE)")
        print("="*80)
        print(f"Method: K-means clustering + Mahalanobis distance to nearest sub-cluster")
        print(f"Threshold: {self.config.CENTROID_FILTER_PERCENTILE}th percentile of original min distances")
        print("-" * 80)
        
        filtered_pca = {}
        
        for class_name in class_names:
            X_synthetic = synthetic_pca[class_name]
            
            if len(X_synthetic) == 0:
                filtered_pca[class_name] = X_synthetic
                print(f"{class_name:<40} No samples to filter")
                continue
            
            # Get threshold
            threshold = class_stats.get_threshold(class_name)
            
            # Compute Mahalanobis distance for each synthetic sample
            mahal_distances = class_stats.compute_min_mahalanobis_distance(
                X_synthetic, class_name
            )
            
            # Filter by threshold
            keep_mask = mahal_distances <= threshold
            X_filtered = X_synthetic[keep_mask]
            filtered_pca[class_name] = X_filtered
            
            kept_pct = 100 * len(X_filtered) / len(X_synthetic) if len(X_synthetic) > 0 else 0
            print(f"{class_name:<40} {len(X_filtered):>5} / {len(X_synthetic):>5} "
                  f"({kept_pct:>5.1f}% kept)")
        
        print("="*80)
        
        return filtered_pca
    
    def _get_filter_method(self, filter_name: str):
        """
        Map filter name to corresponding method.
        
        Args:
            filter_name: Name of the filter ('geometric_margin', 'centroid_distance', 'none')
            
        Returns:
            Filter method or None if 'none'
        """
        filter_map = {
            'geometric_margin': self.filter_by_geometric_margin,
            'centroid_distance': self.filter_by_centroid_distance,
            'none': None
        }
        
        if filter_name not in filter_map:
            raise ValueError(
                f"Unknown filter name: '{filter_name}'. "
                f"Valid options: {list(filter_map.keys())}"
            )
        
        return filter_map[filter_name]
    
    def apply_all_filters(
        self,
        synthetic_pca: Dict[str, np.ndarray],
        pooled_data: np.ndarray,
        pooled_labels: np.ndarray,
        class_stats: ClassStatistics,
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Apply modular filtering pipeline with configurable stages.
        
        Pipeline (modular):
        - Stage 1: Classifier confidence filtering (ALWAYS APPLIED - mandatory)
        - Stage 2: Configurable via FILTER_STAGE_2 in config
        - Stage 3: Configurable via FILTER_STAGE_3 in config
        
        Available filters:
        - 'geometric_margin': Distance-based class separation
        - 'centroid_distance': Mahalanobis distance to sub-cluster centroids
        - 'none': Skip this stage
        
        Args:
            synthetic_pca: Dict[class_name -> synthetic samples in PCA space]
            pooled_data: All pooled data for neighbor search
            pooled_labels: Labels for pooled data
            class_stats: ClassStatistics from original data
            class_names: List of class names (sorted)
            
        Returns:
            filtered_pca: Dict[class_name -> filtered synthetic samples after all stages]
        """
        print("\n" + "="*80)
        print("MODULAR QUALITY FILTERING PIPELINE")
        print("="*80)
        print(f"Stage 1 (MANDATORY): Classifier Confidence")
        print(f"Stage 2 (CONFIG):    {self.config.FILTER_STAGE_2}")
        print(f"Stage 3 (CONFIG):    {self.config.FILTER_STAGE_3}")
        print("="*80)
        
        # Stage 1: Classifier confidence (ALWAYS APPLIED)
        current_result = self.filter_by_classifier_confidence(synthetic_pca, class_names)
        
        # Stage 2: Configurable
        stage2_filter = self._get_filter_method(self.config.FILTER_STAGE_2)
        if stage2_filter is not None:
            if self.config.FILTER_STAGE_2 == 'geometric_margin':
                current_result = stage2_filter(
                    current_result, pooled_data, pooled_labels, class_names
                )
            elif self.config.FILTER_STAGE_2 == 'centroid_distance':
                current_result = stage2_filter(
                    current_result, class_stats, class_names
                )
        else:
            print("\n" + "="*80)
            print("STAGE 2: SKIPPED (configured as 'none')")
            print("="*80)
        
        # Stage 3: Configurable
        stage3_filter = self._get_filter_method(self.config.FILTER_STAGE_3)
        if stage3_filter is not None:
            if self.config.FILTER_STAGE_3 == 'geometric_margin':
                current_result = stage3_filter(
                    current_result, pooled_data, pooled_labels, class_names
                )
            elif self.config.FILTER_STAGE_3 == 'centroid_distance':
                current_result = stage3_filter(
                    current_result, class_stats, class_names
                )
        else:
            print("\n" + "="*80)
            print("STAGE 3: SKIPPED (configured as 'none')")
            print("="*80)
        
        # Print summary
        print("\n" + "="*80)
        print("FILTERING SUMMARY")
        print("="*80)
        for class_name in class_names:
            original = len(synthetic_pca[class_name])
            final = len(current_result[class_name])
            overall_pct = 100 * final / original if original > 0 else 0
            print(f"{class_name:<40} {final:>5} / {original:>5} ({overall_pct:>5.1f}% overall)")
        print("="*80)
        
        return current_result
