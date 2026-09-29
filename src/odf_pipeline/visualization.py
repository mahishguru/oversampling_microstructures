"""
Visualization module for generating comprehensive plots.
Includes PCA projections, per-class plots, and distance distributions.
"""

import os
from typing import Dict
import numpy as np

# Force non-interactive backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection registered)

from .config import Config
from .utils import ClassStatistics


class Visualizer:
    """Generates all visualization plots for the pipeline."""
    
    def __init__(self, config: Config):
        """
        Initialize Visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.colors = None
    
    def _get_colors(self, n_classes: int) -> np.ndarray:
        """Get color palette for classes."""
        if self.colors is None or len(self.colors) != n_classes:
            self.colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        return self.colors
    
    def plot_original_data(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        class_names: list
    ):
        """
        Plot original real data in first 2 PCA dimensions.
        
        Args:
            X_pca_by_class: Dict[class_name -> samples in PCA space]
            class_names: List of class names (sorted)
        """
        print("\nGenerating visualization: Original data (2D PCA)...")
        
        colors = self._get_colors(len(class_names))
        
        plt.figure(figsize=self.config.VIZ_FIGSIZE_MEDIUM)
        
        for i, class_name in enumerate(class_names):
            X_class = X_pca_by_class[class_name]
            plt.scatter(X_class[:, 0], X_class[:, 1],
                       c=[colors[i]], label=class_name, s=100, alpha=0.7, 
                       edgecolors='k')
        
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title('Real ODF Harmonics Data (First 2 Principal Components)', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.config.VISUALIZATION_DIR, 
                                '01_real_data_pca2d.png')
        plt.savefig(save_path, dpi=self.config.VIZ_DPI)
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")

        # 3D PCA scatter (first three components)
        if next(iter(X_pca_by_class.values())).shape[1] >= 3:
            fig = plt.figure(figsize=self.config.VIZ_FIGSIZE_MEDIUM)
            ax = fig.add_subplot(111, projection='3d')
            for i, class_name in enumerate(class_names):
                X_class = X_pca_by_class[class_name]
                ax.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2],
                           c=[colors[i]], label=class_name, s=60, alpha=0.7,
                           edgecolors='none')
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_zlabel('PC3', fontsize=12)
            ax.set_title('Real Data PCA Scatter (3D)', fontsize=14, fontweight='bold')
            ax.view_init(elev=25, azim=40)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            save_path_3d = os.path.join(self.config.VISUALIZATION_DIR,
                                        '01_real_data_pca3d.png')
            plt.tight_layout()
            plt.savefig(save_path_3d, dpi=self.config.VIZ_DPI, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {save_path_3d}")

            # Isometric/highlighted view of the same data
            fig = plt.figure(figsize=self.config.VIZ_FIGSIZE_MEDIUM)
            ax = fig.add_subplot(111, projection='3d')
            for i, class_name in enumerate(class_names):
                X_class = X_pca_by_class[class_name]
                ax.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2],
                           c=[colors[i]], label=class_name, s=60, alpha=0.7,
                           edgecolors='none')
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_zlabel('PC3', fontsize=12)
            ax.set_title('Real Data PCA Scatter (Isometric View)',
                         fontsize=14, fontweight='bold')
            ax.view_init(elev=30, azim=135)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            save_path_iso = os.path.join(self.config.VISUALIZATION_DIR,
                                         '01_real_data_pca3d_isometric.png')
            plt.tight_layout()
            plt.savefig(save_path_iso, dpi=self.config.VIZ_DPI, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {save_path_iso}")
    
    def plot_filtered_comparison(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        filtered_pca: Dict[str, np.ndarray],
        class_stats: ClassStatistics,
        class_names: list
    ):
        """
        Generate 4-panel comparison plot of real vs synthetic data.
        
        Args:
            X_pca_by_class: Original data in PCA space
            filtered_pca: Filtered synthetic data in PCA space
            class_stats: ClassStatistics object
            class_names: List of class names (sorted)
        """
        print("\nGenerating visualization: 4-panel filtered comparison...")
        
        colors = self._get_colors(len(class_names))
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.VIZ_FIGSIZE_LARGE)
        axes = axes.flatten()
        
        # Panel 1: Real data only (darker, larger stars)
        ax = axes[0]
        for i, class_name in enumerate(class_names):
            X_class = X_pca_by_class[class_name]
            darker_color = colors[i] * 0.7
            ax.scatter(X_class[:, 0], X_class[:, 1],
                      c=[darker_color], label=class_name, s=200, alpha=0.9,
                      edgecolors='k', linewidths=2, marker='*')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('Original Real Data (Darker Stars)', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Panel 2: Synthetic data only
        ax = axes[1]
        for i, class_name in enumerate(class_names):
            X_synth = filtered_pca[class_name]
            if len(X_synth) > 0:
                ax.scatter(X_synth[:, 0], X_synth[:, 1],
                          c=[colors[i]], label=class_name, s=50, alpha=0.6,
                          edgecolors='none')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('Filtered Synthetic Data', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Panel 3: Real + Synthetic overlay
        ax = axes[2]
        for i, class_name in enumerate(class_names):
            # Original (darker, larger stars)
            X_class = X_pca_by_class[class_name]
            darker_color = colors[i] * 0.7
            ax.scatter(X_class[:, 0], X_class[:, 1],
                      c=[darker_color], s=200, alpha=0.95,
                      edgecolors='k', linewidths=2, marker='*',
                      label=f'{class_name} (original)')
            
            # Synthetic (lighter, smaller circles)
            X_synth = filtered_pca[class_name]
            if len(X_synth) > 0:
                ax.scatter(X_synth[:, 0], X_synth[:, 1],
                          c=[colors[i]], s=30, alpha=0.4,
                          edgecolors='none', label=f'{class_name} (synthetic)')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('Combined: Original (Large Dark Stars) + Synthetic (Small Light Dots)',
                    fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        
        # Panel 4: Distance distributions
        ax = axes[3]
        for i, class_name in enumerate(class_names):
            threshold = class_stats.get_threshold(class_name)
            
            # Original distances
            orig_distances = class_stats.get_distances(class_name)
            
            # Synthetic distances
            X_synth = filtered_pca[class_name]
            if len(X_synth) > 0:
                synth_distances = class_stats.compute_min_mahalanobis_distance(
                    X_synth,
                    class_name
                )
                
                # Plot histograms
                ax.hist(orig_distances, bins=20, alpha=0.5, color=colors[i]*0.7,
                       label=f'{class_name} (orig)', edgecolor='black', linewidth=1.5)
                ax.hist(synth_distances, bins=20, alpha=0.3, color=colors[i],
                       label=f'{class_name} (synth)')
                
                # Mark threshold
                ax.axvline(threshold, color=colors[i], linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Minimum Subcluster Mahalanobis Distance', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distance Distributions (Dashed line = {self.config.CENTROID_FILTER_PERCENTILE}th percentile)',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.VISUALIZATION_DIR,
                                '02_filtered_synthetic_vs_real.png')
        plt.savefig(save_path, dpi=self.config.VIZ_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def plot_per_class_details(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        filtered_pca: Dict[str, np.ndarray],
        class_stats: ClassStatistics,
        class_names: list
    ):
        """
        Generate detailed per-class plots with centroid and threshold circle.
        
        Args:
            X_pca_by_class: Original data in PCA space
            filtered_pca: Filtered synthetic data in PCA space
            class_stats: ClassStatistics object
            class_names: List of class names (sorted)
        """
        print("\nGenerating per-class detailed visualizations...")
        
        colors = self._get_colors(len(class_names))
        
        for i, class_name in enumerate(class_names):
            fig = plt.figure(figsize=self.config.VIZ_FIGSIZE_SMALL)
            ax = fig.add_subplot(111)
            
            # Original points (darker, larger stars)
            X_class = X_pca_by_class[class_name]
            darker_color = colors[i] * 0.7
            ax.scatter(X_class[:, 0], X_class[:, 1],
                      c=[darker_color], s=250, alpha=0.95,
                      edgecolors='k', linewidths=2.5, marker='*',
                      label='Original', zorder=3)
            
            # Synthetic points (lighter, smaller circles)
            X_synth = filtered_pca[class_name]
            if len(X_synth) > 0:
                ax.scatter(X_synth[:, 0], X_synth[:, 1],
                          c=[colors[i]], s=40, alpha=0.5,
                          edgecolors='none', label='Synthetic', zorder=2)
            
            # Draw centroid and threshold circle
            centroid = class_stats.get_centroid(class_name)
            threshold = class_stats.get_threshold(class_name)
            
            ax.scatter([centroid[0]], [centroid[1]], c='red', s=300,
                      marker='X', edgecolors='black', linewidths=2,
                      label='Centroid', zorder=4)
            
            ax.text(
                0.02,
                0.98,
                f'Mahalanobis threshold: {threshold:.2f}',
                transform=ax.transAxes,
                fontsize=10,
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none')
            )
            
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title(f'{class_name}\nOriginal: {len(X_class)}, Synthetic: {len(X_synth)}',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            save_path = os.path.join(self.config.VISUALIZATION_DIR,
                                    f'03_class_{class_name}.png')
            plt.savefig(save_path, dpi=self.config.VIZ_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {save_path}")
    
    def plot_final_comparison(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        filtered_pca: Dict[str, np.ndarray],
        class_stats: ClassStatistics,
        class_names: list
    ):
        """
        Generate final 4-panel comparison plot.
        
        Args:
            X_pca_by_class: Original data in PCA space
            filtered_pca: Filtered synthetic data in PCA space
            class_stats: ClassStatistics object
            class_names: List of class names (sorted)
        """
        print("\nGenerating visualization: Final comparison...")
        
        colors = self._get_colors(len(class_names))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot 1: All real data
        ax = axes[0]
        for i, class_name in enumerate(class_names):
            X_class = X_pca_by_class[class_name]
            ax.scatter(X_class[:, 0], X_class[:, 1],
                      c=[colors[i]], label=class_name, s=80, alpha=0.7,
                      edgecolors='k')
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('Real Data Only', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 2: All synthetic data
        ax = axes[1]
        for i, class_name in enumerate(class_names):
            X_synth = filtered_pca[class_name]
            if len(X_synth) > 0:
                ax.scatter(X_synth[:, 0], X_synth[:, 1],
                          c=[colors[i]], label=class_name, s=20, alpha=0.4)
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('Synthetic Data Only', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 3: Real + Synthetic combined
        ax = axes[2]
        for i, class_name in enumerate(class_names):
            X_real = X_pca_by_class[class_name]
            X_synth = filtered_pca[class_name]
            
            ax.scatter(X_real[:, 0], X_real[:, 1],
                      c=[colors[i]], s=100, alpha=0.9, edgecolors='k', linewidths=2,
                      label=f'{class_name} (real)')
            if len(X_synth) > 0:
                ax.scatter(X_synth[:, 0], X_synth[:, 1],
                          c=[colors[i]], s=20, alpha=0.3, marker='x')
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('Real (dots) + Synthetic (x)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 4: Distance distributions
        ax = axes[3]
        for i, class_name in enumerate(class_names):
            real_dists = class_stats.get_distances(class_name)
            X_synth = filtered_pca[class_name]
            
            if len(X_synth) > 0:
                synth_dists = class_stats.compute_min_mahalanobis_distance(
                    X_synth,
                    class_name
                )
                ax.hist(real_dists, bins=20, alpha=0.5, label=f'{class_name} (real)',
                       color=colors[i], edgecolor='black')
                ax.hist(synth_dists, bins=20, alpha=0.3, label=f'{class_name} (synth)',
                       color=colors[i], linestyle='--')
        ax.set_xlabel('Minimum Subcluster Mahalanobis Distance', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distance Distributions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.VISUALIZATION_DIR,
                                '04_final_comparison.png')
        plt.savefig(save_path, dpi=self.config.VIZ_DPI)
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def generate_all_visualizations(
        self,
        X_pca_by_class: Dict[str, np.ndarray],
        filtered_pca: Dict[str, np.ndarray],
        class_stats: ClassStatistics,
        class_names: list
    ):
        """
        Generate all visualizations.
        
        Args:
            X_pca_by_class: Original data in PCA space
            filtered_pca: Filtered synthetic data in PCA space
            class_stats: ClassStatistics object
            class_names: List of class names (sorted)
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        self.plot_original_data(X_pca_by_class, class_names)
        self.plot_filtered_comparison(X_pca_by_class, filtered_pca, 
                                      class_stats, class_names)
        self.plot_per_class_details(X_pca_by_class, filtered_pca, 
                                    class_stats, class_names)
        self.plot_final_comparison(X_pca_by_class, filtered_pca, 
                                   class_stats, class_names)
        
        print("\n" + "="*80)
        print("All visualizations generated successfully!")
        print("="*80)
