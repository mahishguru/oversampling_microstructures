"""
Synthetic sample generation using sequential non-updating ADASYN + optional Contextual Adjustment.

Augments each class by ADASYN_OVERSAMPLE_RATIO (e.g., 10x) for data augmentation.
The pipeline generates synthetic samples through:
1. Sequential per-class ADASYN with real other-class context (non-updating)
2. Optional contextual adjustment (cross-class nudging)

Each class is processed independently using the same original dataset,
ensuring order-independent and reproducible results with deterministic per-class seeding.
"""

from typing import Dict
import hashlib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from imblearn.over_sampling import ADASYN

from .config import Config


class HybridSampler:
    """
    Sequential non-updating ADASYN-based synthetic sample generator with optional contextual adjustment.
    
    Generates synthetic samples using:
    - Sequential per-class ADASYN with real other-class context (non-updating)
    - Each class uses the full original dataset but does not append synthetic samples between runs
    - Deterministic per-class seeding (order-independent)
    - Optional contextual adjustment for cross-class boundary awareness
    """
    
    def __init__(self, config: Config):
        """
        Initialize HybridSampler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        np.random.seed(config.RANDOM_SEED)
    
    @staticmethod
    def _per_class_seed(base_seed: int, cls: int) -> int:
        """
        Generate deterministic per-class seed independent of processing order.
        
        Uses hash of class label to ensure same seed regardless of order classes are processed.
        
        Args:
            base_seed: Base random seed
            cls: Class label (integer)
            
        Returns:
            Deterministic seed for this class (32-bit positive integer)
        """
        cls_bytes = str(cls).encode('utf-8')
        h = hashlib.sha256(cls_bytes).hexdigest()
        # Take lower 8 hex digits -> int, mod to keep within safe range
        cls_hash = int(h[:8], 16) % 1_000_000
        # Use modulo arithmetic to prevent overflow
        # 2**31 - 1 is the max value for numpy's RandomState seed
        seed = (int(base_seed) % 1_000_000 + cls_hash) % (2**31 - 1)
        return max(0, seed)  # Ensure non-negative
    
    @staticmethod
    def _extract_new_samples(
        X_old: np.ndarray,
        y_old: np.ndarray,
        X_res: np.ndarray,
        y_res: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Robustly determine which samples in (X_res, y_res) are synthetic relative to (X_old, y_old).
        
        For each class:
        - Compute distance from each resampled point to every original point of the same class
        - Treat the resampled points with the smallest distances (up to original_count) as originals
        - Remaining resampled points are synthetic
        
        Uses memory-efficient chunked processing for large datasets to avoid OOM errors.
        
        Args:
            X_old: Original data before ADASYN
            y_old: Original labels before ADASYN
            X_res: Resampled data after ADASYN
            y_res: Resampled labels after ADASYN
            
        Returns:
            Dict {label: np.ndarray of new synthetic samples for that label}
        """
        new_by_class = {}
        labels_res = np.unique(y_res)
        
        for lab in labels_res:
            X_res_lab = X_res[y_res == lab]
            X_old_lab = X_old[y_old == lab]
            n_old = X_old_lab.shape[0]
            n_res = X_res_lab.shape[0]
            
            if n_old == 0:
                # Class didn't exist before -> all are new
                new_by_class[lab] = X_res_lab.copy()
                continue
            
            if n_res <= n_old:
                # Edge case: resampled has fewer or equal samples than original
                # This shouldn't happen with ADASYN, but handle gracefully
                new_by_class[lab] = np.empty((0, X_res.shape[1]))
                continue
            
            # Memory-efficient distance computation using chunking
            # Process in chunks to avoid creating huge distance matrices
            chunk_size = min(1000, n_res)  # Process 1000 samples at a time
            min_dist = np.zeros(n_res)
            
            for start_idx in range(0, n_res, chunk_size):
                end_idx = min(start_idx + chunk_size, n_res)
                chunk = X_res_lab[start_idx:end_idx]
                # Compute distances for this chunk
                d_chunk = pairwise_distances(chunk, X_old_lab, metric='euclidean')
                min_dist[start_idx:end_idx] = d_chunk.min(axis=1)
            
            # Sort resampled points by distance to nearest original
            sorted_idx = np.argsort(min_dist)
            
            # Mark up to n_old closest resampled as originals, rest as synthetic
            n_mark_original = min(n_old, n_res)
            mark_original_indices = sorted_idx[:n_mark_original]
            mark_original_mask = np.zeros(n_res, dtype=bool)
            mark_original_mask[mark_original_indices] = True
            
            synthetic_indices = np.where(~mark_original_mask)[0]
            if synthetic_indices.size == 0:
                new_by_class[lab] = np.empty((0, X_res.shape[1]))
            else:
                new_by_class[lab] = X_res_lab[synthetic_indices]
        
        return new_by_class
    
    def _sequential_adasyn_per_class(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        target_class: int,
        desired_total: int
    ) -> np.ndarray:
        """
        Generate synthetic samples for one class using ADASYN on full dataset.
        
        Uses real other-class context (not dummy classes) for boundary detection.
        Does NOT append synthetic samples back to dataset (non-updating).
        
        Pipeline:
        1. Run ADASYN on full dataset (X_all, y_all) with sampling_strategy={target_class: desired_total}
        2. Extract only the newly generated synthetic samples for target_class
        3. Optionally apply contextual nudge if USE_CONTEXTUAL_NUDGE is True
        4. Return synthetic samples
        
        Args:
            X_all: (n_samples, n_components) all pooled data in PCA space
            y_all: (n_samples,) all class labels
            target_class: Class label to generate samples for
            desired_total: Total count desired after resampling (original + synthetic)
            
        Returns:
            X_synthetic: (n_generated, n_components) newly generated synthetic samples
        
        Raises:
            ValueError: If input validation fails or insufficient samples
            RuntimeError: If ADASYN generation fails
        """
        # Input validation
        if X_all.shape[0] != y_all.shape[0]:
            raise ValueError(
                f"X_all and y_all shape mismatch: X_all has {X_all.shape[0]} samples, "
                f"y_all has {y_all.shape[0]} samples."
            )
        
        # Get current count of this class
        n_current = int(np.sum(y_all == target_class))
        
        if desired_total <= n_current:
            # Error: desired total is not greater than current count
            raise ValueError(
                f"Cannot generate synthetic samples for class {target_class}: "
                f"desired_total ({desired_total}) must be greater than current count ({n_current}). "
                f"Check ADASYN_OVERSAMPLE_RATIO configuration."
            )
        
        if n_current < 2:
            # Not enough samples for ADASYN - raise error instead of fallback
            raise ValueError(
                f"Cannot run ADASYN for class {target_class}: only {n_current} sample(s) available. "
                f"ADASYN requires at least 2 samples per class. "
                f"Please ensure sufficient data or remove classes with insufficient samples."
            )
        
        # Generate deterministic per-class seed (order-independent)
        seed = self._per_class_seed(self.config.RANDOM_SEED, target_class)
        
        # Choose k robustly: ADASYN requires n_neighbors < n_minority
        k = min(self.config.ADASYN_K_NEIGHBORS, max(1, n_current - 1))
        
        # ====================================================================
        # STAGE 1: ADASYN Generation with Real Other-Class Context
        # ====================================================================
        try:
            adasyn = ADASYN(
                sampling_strategy={target_class: desired_total},
                n_neighbors=k,
                random_state=seed
            )
            X_res, y_res = adasyn.fit_resample(X_all, y_all)
            
            # Extract only the newly generated synthetic samples for this class
            new_dict = self._extract_new_samples(X_all, y_all, X_res, y_res)
            candidates = new_dict.get(target_class, np.empty((0, X_all.shape[1])))
            
        except RuntimeError as e:
            # ADASYN can fail when minority samples have no majority neighbors
            # Fall back to SMOTE which doesn't have this constraint
            if "neigbours belong to the majority class" in str(e) or "NaN" in str(e):
                print(f"    ⚠️  ADASYN failed (neighbor distribution issue), falling back to SMOTE")
                from imblearn.over_sampling import SMOTE
                
                smote = SMOTE(
                    sampling_strategy={target_class: desired_total},
                    k_neighbors=k,
                    random_state=seed
                )
                X_res, y_res = smote.fit_resample(X_all, y_all)
                
                # Extract newly generated samples
                new_dict = self._extract_new_samples(X_all, y_all, X_res, y_res)
                candidates = new_dict.get(target_class, np.empty((0, X_all.shape[1])))
            else:
                # Re-raise other types of errors with context
                raise RuntimeError(
                    f"ADASYN failed for class {target_class} (n_samples={n_current}, "
                    f"desired_total={desired_total}, n_neighbors={k}). "
                    f"Original error: {str(e)}"
                ) from e
        
        if len(candidates) == 0:
            raise RuntimeError(
                f"ADASYN generated 0 synthetic samples for class {target_class}. "
                f"This may indicate a problem with the data or parameters."
            )
        
        # ====================================================================
        # STAGE 2: Optional Contextual Adjustment (Cross-Class Aware)
        # ====================================================================
        if not self.config.USE_CONTEXTUAL_NUDGE:
            # Return ADASYN samples directly without contextual adjustment
            return candidates
        
        # Apply contextual nudge
        X_other = X_all[y_all != target_class]  # Other class samples
        
        if len(X_other) == 0:
            # No other classes, return candidates as-is
            return candidates
        
        # Generate context vectors based on mode
        if self.config.CONTEXT_MODE == 'other_centroid':
            # Use global centroid of all other classes
            ctx_vecs = np.tile(X_other.mean(axis=0), (len(candidates), 1))
        elif self.config.CONTEXT_MODE == 'nearest_other':
            # Use nearest other-class sample
            nbrs_other = NearestNeighbors(n_neighbors=1).fit(X_other)
            _, idxs = nbrs_other.kneighbors(candidates)
            ctx_vecs = X_other[idxs.flatten()]
        elif self.config.CONTEXT_MODE == 'nearest_other_centroid':
            # Use mean of k nearest other-class samples
            k_ctx = min(3, len(X_other))
            nbrs_other_many = NearestNeighbors(n_neighbors=k_ctx).fit(X_other)
            _, idxs = nbrs_other_many.kneighbors(candidates)
            # Vectorized mean computation
            ctx_vecs = np.array([X_other[idxs[i]].mean(axis=0) for i in range(len(idxs))])
        else:
            # Fallback to global other centroid
            ctx_vecs = np.tile(X_other.mean(axis=0), (len(candidates), 1))
        
        # Apply contextual nudge to all candidates (vectorized)
        # x_adjusted = x_adasyn + alpha * (x_ctx - x_adasyn)
        #            = x_adasyn * (1 - alpha) + x_ctx * alpha
        synthetic_samples = candidates * (1 - self.config.CONTEXTUAL_ALPHA) + \
                           ctx_vecs * self.config.CONTEXTUAL_ALPHA
        
        return synthetic_samples
    
    def _apply_contextual_nudge(
        self,
        candidates: np.ndarray,
        X_other: np.ndarray
    ) -> np.ndarray:
        """
        Nudge candidates slightly toward other-class context (boundary awareness).

        Shared by the ADASYN and distribution-fill generators. Returns candidates
        unchanged when contextual nudging is disabled or no other-class data exists.
        """
        if not self.config.USE_CONTEXTUAL_NUDGE or len(X_other) == 0 or len(candidates) == 0:
            return candidates

        if self.config.CONTEXT_MODE == 'other_centroid':
            ctx_vecs = np.tile(X_other.mean(axis=0), (len(candidates), 1))
        elif self.config.CONTEXT_MODE == 'nearest_other':
            nbrs_other = NearestNeighbors(n_neighbors=1).fit(X_other)
            _, idxs = nbrs_other.kneighbors(candidates)
            ctx_vecs = X_other[idxs.flatten()]
        elif self.config.CONTEXT_MODE == 'nearest_other_centroid':
            k_ctx = min(3, len(X_other))
            nbrs_other_many = NearestNeighbors(n_neighbors=k_ctx).fit(X_other)
            _, idxs = nbrs_other_many.kneighbors(candidates)
            ctx_vecs = X_other[idxs].mean(axis=1)
        else:
            ctx_vecs = np.tile(X_other.mean(axis=0), (len(candidates), 1))

        return candidates * (1 - self.config.CONTEXTUAL_ALPHA) + ctx_vecs * self.config.CONTEXTUAL_ALPHA

    def _subcluster_labels(self, X_class: np.ndarray, seed: int) -> np.ndarray:
        """
        Split a class into subclusters so multi-modal classes are filled per-mode.

        Returns an integer label per point. Classes smaller than the configured
        minimum stay as a single cluster.
        """
        n = len(X_class)
        if n < self.config.DISTRIBUTION_MIN_SUBCLUSTER:
            return np.zeros(n, dtype=int)
        n_clusters = min(3, max(1, n // 3))
        if n_clusters <= 1:
            return np.zeros(n, dtype=int)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(X_class)

    def _barycentric_samples(
        self,
        points: np.ndarray,
        n_generate: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Fill the interior of a point set via Dirichlet-weighted barycentric mixing.

        Each synthetic point is a random convex combination of m=BARYCENTRIC_POINTS
        same-cluster points, weights drawn from Dirichlet(alpha). This populates the
        convex interior of the cluster rather than just the 1D edges between pairs.
        """
        n_pts = len(points)
        if n_pts == 0 or n_generate <= 0:
            return np.empty((0, points.shape[1]))
        if n_pts == 1:
            return np.repeat(points, n_generate, axis=0)

        m = min(self.config.BARYCENTRIC_POINTS, n_pts)
        replace = n_pts < m
        # Index matrix (n_generate, m): which points are mixed for each sample
        idx = np.array([
            rng.choice(n_pts, size=m, replace=replace) for _ in range(n_generate)
        ])
        weights = rng.dirichlet(np.full(m, self.config.DIRICHLET_ALPHA), size=n_generate)
        gathered = points[idx]  # (n_generate, m, d)
        return np.einsum('nm,nmd->nd', weights, gathered)

    def _gaussian_samples(
        self,
        points: np.ndarray,
        n_generate: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Sample from a shrinkage-covariance Gaussian fit of a point set.

        cov scale = GAUSSIAN_COV_SCALE: 1.0 fills the ellipsoid interior,
        >1 produces controlled extrapolation into the distribution tails.
        """
        n_pts = len(points)
        if n_pts == 0 or n_generate <= 0:
            return np.empty((0, points.shape[1]))
        mean = points.mean(axis=0)
        if n_pts < 2:
            return np.repeat(mean[None, :], n_generate, axis=0)
        cov = LedoitWolf().fit(points).covariance_
        cov = cov * (self.config.GAUSSIAN_COV_SCALE ** 2)
        cov += np.eye(cov.shape[0]) * 1e-9  # numerical PD guard
        return rng.multivariate_normal(mean, cov, size=n_generate)

    def _bridge_samples(
        self,
        subcluster_points: list,
        n_generate: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Fill the space BETWEEN subclusters via cross-mode convex mixing.

        Each bridge sample is a Dirichlet-weighted combination of one random point
        drawn from each of 2..n_sub distinct subclusters, so the inter-mode gaps
        (which per-subcluster filling leaves empty) get populated. This restores the
        global manifold continuity that pure per-mode filling loses.
        """
        non_empty = [p for p in subcluster_points if len(p) > 0]
        n_sub = len(non_empty)
        if n_sub < 2 or n_generate <= 0:
            return np.empty((0, non_empty[0].shape[1])) if non_empty else np.empty((0, 0))

        dim = non_empty[0].shape[1]
        out = np.empty((n_generate, dim))
        for i in range(n_generate):
            k = rng.randint(2, n_sub + 1)  # mix 2..n_sub distinct modes
            chosen = rng.choice(n_sub, size=k, replace=False)
            verts = np.array([
                non_empty[c][rng.randint(len(non_empty[c]))] for c in chosen
            ])
            w = rng.dirichlet(np.full(k, self.config.DIRICHLET_ALPHA))
            out[i] = w @ verts
        return out

    def _distribution_fill_per_class(
        self,
        X_real: np.ndarray,
        X_fill: np.ndarray,
        X_other: np.ndarray,
        n_generate: int,
        seed: int
    ) -> np.ndarray:
        """
        Generate class-aware, interior-filling candidates for one class.

        Modes are defined from the REAL points (so augmentation cannot invent fake
        modes), then the dense fill basis (real + jittered + augmented) is assigned
        to its nearest mode to enrich each subcluster's covariance and interior.
        Within each mode we blend Dirichlet barycentric interpolation (convex
        interior) with Gaussian sampling (ellipsoid fill + controlled extrapolation);
        a BRIDGE_FRACTION share is generated as cross-mode bridges to fill the
        inter-mode gaps. Local allocation is proportional to real subcluster size.
        """
        if n_generate <= 0 or len(X_real) == 0:
            return np.empty((0, X_real.shape[1]))

        rng = np.random.RandomState(seed)

        # 1) Define modes from the real points.
        real_labels = self._subcluster_labels(X_real, seed)
        unique = np.unique(real_labels)
        centroids = np.array([X_real[real_labels == c].mean(axis=0) for c in unique])

        # 2) Assign the dense basis to nearest real mode (enriches covariance/interior).
        if X_fill is None or len(X_fill) == 0:
            X_fill = X_real
        fill_assign = np.argmin(
            ((X_fill[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        subcluster_points = []
        for k in range(len(unique)):
            pts = X_fill[fill_assign == k]
            if len(pts) == 0:  # backfill empty modes with their real points
                pts = X_real[real_labels == unique[k]]
            subcluster_points.append(pts)

        # 3) Split budget: cross-mode bridges + per-mode local fill.
        sizes = np.array([int(np.sum(real_labels == c)) for c in unique])
        bridge_frac = self.config.BRIDGE_FRACTION if len(unique) >= 2 else 0.0
        n_bridge = int(round(n_generate * bridge_frac))
        n_local = n_generate - n_bridge

        alloc = np.floor(sizes / sizes.sum() * n_local).astype(int)
        remainder = n_local - int(alloc.sum())
        for i in np.argsort(-sizes)[:max(0, remainder)]:
            alloc[i] += 1

        g_frac = self.config.GAUSSIAN_FRACTION
        parts = []
        for k, n_sub in enumerate(alloc):
            if n_sub <= 0:
                continue
            pts = subcluster_points[k]
            n_gauss = int(round(n_sub * g_frac))
            n_bary = n_sub - n_gauss
            if n_bary > 0:
                parts.append(self._barycentric_samples(pts, n_bary, rng))
            if n_gauss > 0:
                parts.append(self._gaussian_samples(pts, n_gauss, rng))

        if n_bridge > 0:
            bridge = self._bridge_samples(subcluster_points, n_bridge, rng)
            if len(bridge) > 0:
                parts.append(bridge)

        if not parts:
            return np.empty((0, X_real.shape[1]))
        candidates = np.vstack(parts)
        return self._apply_contextual_nudge(candidates, X_other)

    def augment_classes(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        jittered_pca: Dict[str, np.ndarray],
        augmented_pca: Dict[str, np.ndarray],
        class_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Generate target-sized candidate sets using sequential non-updating ADASYN.
        
        Each class produces a pre-filter candidate pool based on the configured
        per-class synthetic target rather than on the already-augmented pool size:
        - Sequential per-class ADASYN with real other-class context
        - Non-updating: each class uses the same original pooled dataset
        - Order-independent deterministic seeding
        - Optional contextual nudge (controlled by USE_CONTEXTUAL_NUDGE)
        
        Args:
            X_pca_by_class: Original data in PCA space per class
            jittered_pca: Jittered samples per class
            augmented_pca: Augmented samples per class
            class_names: List of class names (sorted)
            
        Returns:
            result_pca: Dict[class_name -> ADASYN candidate samples in PCA space]
        """
        print("\n" + "="*80)
        print("SEQUENTIAL NON-UPDATING ADASYN CANDIDATE GENERATION")
        print("="*80)
        
        # Pool all augmented samples (this is the base dataset for ADASYN)
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
        
        print(f"\nPooled augmented data (base for ADASYN):")
        for i, class_name in enumerate(class_names):
            count = np.sum(y_pooled == i)
            print(f"  {class_name:<40} {count:>6} samples")
        
        # Display configuration
        contextual_status = "ENABLED" if self.config.USE_CONTEXTUAL_NUDGE else "DISABLED"
        print(f"\nConfiguration:")
        print(f"  Generation Mode: {self.config.GENERATION_MODE}")
        print(f"  Target Synthetic Per Class: {self.config.TARGET_SYNTHETIC_PER_CLASS}")
        print(f"  Candidate Multiplier: {self.config.ADASYN_CANDIDATE_MULTIPLIER}x")
        print(f"  Contextual Nudge: {contextual_status}")
        if self.config.USE_CONTEXTUAL_NUDGE:
            print(f"    - Alpha: {self.config.CONTEXTUAL_ALPHA}")
            print(f"    - Mode: {self.config.CONTEXT_MODE}")
        print(f"  Sequential Strategy: Non-updating (each class uses same base dataset)")
        
        # Generate candidate synthetic samples for each class sequentially
        print(f"\nGenerating candidate pools for target-aware filtering:")
        print("  (Processing order does not affect results - deterministic per-class seeding)")
        print("-" * 80)
        
        result_pca = {}
        
        for i, class_name in enumerate(class_names):
            n_original = len(X_pca_by_class[class_name])
            target_synthetic = self.config.target_synthetic_count(n_original)
            n_synthetic_to_generate = self.config.candidate_synthetic_count(n_original)

            # Get current class size in pooled context data
            n_current = int(np.sum(y_pooled == i))
            
            # Validate that class exists and has samples
            if n_current == 0:
                raise ValueError(
                    f"Class '{class_name}' (index {i}) has 0 samples in pooled data. "
                    f"Cannot generate synthetic samples for empty class."
                )
            
            desired_total = n_current + n_synthetic_to_generate
            
            # This should never happen with positive ADASYN_OVERSAMPLE_RATIO, but check anyway
            if n_synthetic_to_generate <= 0:
                raise ValueError(
                    f"Class '{class_name}': calculated n_synthetic_to_generate={n_synthetic_to_generate} "
                    f"(target={target_synthetic}, candidate_count={n_synthetic_to_generate}). "
                    f"This indicates a configuration error."
                )
            
            print(
                f"  {class_name:<40} Final target: {target_synthetic:>6} | "
                f"Candidates: {n_synthetic_to_generate:>6} | ",
                end='',
                flush=True
            )
            
            # Generate synthetic samples using sequential ADASYN on full dataset
            # Note: X_pooled and y_pooled stay constant (non-updating)
            if self.config.GENERATION_MODE == 'distribution':
                # Blend: optional ADASYN edges + barycentric interior + Gaussian tails.
                # Interior fill uses the real class points (not the radial augmented
                # streaks) so the convex interior + distribution tails get populated.
                seed = self._per_class_seed(self.config.RANDOM_SEED, i)
                X_class = X_pca_by_class[class_name]
                X_other = X_pooled[y_pooled != i]

                # Dense fill basis: reuse the already-computed jitter+augment halo so
                # the covariance/interior is estimated from many points, not just the
                # 2-18 bare originals (richer, more diverse fill at no extra cost).
                if self.config.USE_AUGMENTED_FILL_BASIS:
                    X_fill = np.vstack([
                        X_pca_by_class[class_name],
                        jittered_pca[class_name],
                        augmented_pca[class_name],
                    ])
                else:
                    X_fill = X_class

                # 1) ADASYN/SMOTE edge component (boundary-aware, follows neighbours)
                X_edge = np.empty((0, X_pooled.shape[1]))
                n_edge = int(round(n_synthetic_to_generate * self.config.EDGE_FRACTION))
                if n_edge > 0:
                    try:
                        X_edge = self._sequential_adasyn_per_class(
                            X_pooled, y_pooled,
                            target_class=i,
                            desired_total=n_current + n_edge
                        )
                    except (ValueError, RuntimeError):
                        X_edge = np.empty((0, X_pooled.shape[1]))

                # 2) Interior + tails + bridges fill the remainder (guarantees target)
                n_interior = max(0, n_synthetic_to_generate - len(X_edge))
                X_interior = self._distribution_fill_per_class(
                    X_class, X_fill, X_other, n_generate=n_interior, seed=seed
                )
                X_synthetic = np.vstack([X_edge, X_interior]) if len(X_edge) else X_interior
            else:
                X_synthetic = self._sequential_adasyn_per_class(
                    X_pooled,
                    y_pooled,
                    target_class=i,
                    desired_total=desired_total
                )
            
            result_pca[class_name] = X_synthetic
            
            actual_generated = len(X_synthetic)
            achievement = (actual_generated / n_synthetic_to_generate * 100) if n_synthetic_to_generate > 0 else 0
            print(f"Generated: {actual_generated:>6} ({achievement:.1f}%)")
        
        print("="*80)
        print("Candidate generation complete")
        print("Method: Sequential non-updating ADASYN with real other-class context")
        print("="*80)
        
        return result_pca
