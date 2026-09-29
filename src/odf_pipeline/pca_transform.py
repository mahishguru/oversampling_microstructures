"""
PCA transformation module for dimensionality reduction.
Handles StandardScaler + PCA fitting, transforming, and inverse transforming.
"""

import pickle
import os
from typing import Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import Config


class PCATransformer:
    """Handles PCA transformation and inverse transformation with scaling."""
    
    def __init__(self, config: Config):
        """
        Initialize PCATransformer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit StandardScaler and PCA on data, then transform.
        
        Args:
            X: (n_samples, n_features) array of original data
            
        Returns:
            X_pca: (n_samples, n_components) transformed data in PCA space
        """
        print("\n" + "="*80)
        print("FITTING PCA AND STANDARDIZER")
        print("="*80)
        
        # Fit StandardScaler
        self.scaler = StandardScaler()
        X_standardized = self.scaler.fit_transform(X)
        
        print(f"Data standardized.")
        print(f"  Mean: {X_standardized.mean():.10f}")
        print(f"  Std:  {X_standardized.std():.6f}")
        
        # Fit PCA
        self.pca = PCA(n_components=self.config.PCA_COMPONENTS)
        X_pca = self.pca.fit_transform(X_standardized)
        
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        
        print(f"\nPCA fitted with {self.config.PCA_COMPONENTS} components")
        print(f"  Explained variance ratio: {self.cumulative_variance_[-1]:.4f}")
        print(f"  Top 10 component variances: {self.explained_variance_ratio_[:10]}")
        
        return X_pca
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to PCA space.
        
        Args:
            X: (n_samples, n_features) array of original data
            
        Returns:
            X_pca: (n_samples, n_components) transformed data
        """
        if self.scaler is None or self.pca is None:
            raise RuntimeError("PCATransformer must be fitted before transform")
        
        X_standardized = self.scaler.transform(X)
        X_pca = self.pca.transform(X_standardized)
        return X_pca
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Transform data from PCA space back to original space.
        
        Args:
            X_pca: (n_samples, n_components) data in PCA space
            
        Returns:
            X: (n_samples, n_features) data in original space
        """
        if self.scaler is None or self.pca is None:
            raise RuntimeError("PCATransformer must be fitted before inverse_transform")
        
        X_standardized = self.pca.inverse_transform(X_pca)
        X = self.scaler.inverse_transform(X_standardized)
        return X
    
    def save(self, filepath: Optional[str] = None):
        """
        Save PCA and scaler to disk.
        
        Args:
            filepath: Path to save file (default: logs/pca_scaler.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.config.LOGS_DIR, 'pca_scaler.pkl')
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'scaler': self.scaler,
                'explained_variance_ratio': self.explained_variance_ratio_,
                'cumulative_variance': self.cumulative_variance_
            }, f)
        
        print(f"PCA and scaler saved to: {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """
        Load PCA and scaler from disk.
        
        Args:
            filepath: Path to load file (default: logs/pca_scaler.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.config.LOGS_DIR, 'pca_scaler.pkl')
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.pca = data['pca']
        self.scaler = data['scaler']
        self.explained_variance_ratio_ = data.get('explained_variance_ratio')
        self.cumulative_variance_ = data.get('cumulative_variance')
        
        print(f"PCA and scaler loaded from: {filepath}")
    
    def get_pca_summary(self) -> str:
        """
        Get a summary of PCA fitting.
        
        Returns:
            Formatted string with PCA statistics
        """
        if self.pca is None:
            return "PCA not fitted yet"
        
        lines = [
            f"PCA Components: {self.config.PCA_COMPONENTS}",
            f"Explained Variance: {self.cumulative_variance_[-1]:.4f}",
            f"Input Dimension: {self.pca.n_features_in_}",
            f"Output Dimension: {self.pca.n_components_}",
        ]
        return "\n".join(lines)
