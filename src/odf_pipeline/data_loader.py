"""
Data loading module for ODF harmonics files.
Handles file discovery, class extraction, and data organization.
"""

import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

from .config import Config


class DataLoader:
    """Loads and organizes ODF harmonics data by class."""
    
    def __init__(self, config: Config):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.class_data: Dict[str, np.ndarray] = {}
        self.class_files: Dict[str, List[str]] = {}
        self.class_names_sorted: List[str] = []
        
    @staticmethod
    def extract_class_name(filename: str) -> str:
        """
        Extract class name from filename.
        
        Assumes format: {class_name}_{digits}_odf_harmonics.txt
        Returns everything before the first digit.
        
        Args:
            filename: Full path or basename of file
            
        Returns:
            Extracted class name
            
        Example:
            'AZ31_extruded_1_odf_harmonics.txt' -> 'AZ31_extruded'
        """
        base = os.path.basename(filename)
        # Strip expected suffix if present
        if base.endswith('_odf_harmonics.txt'):
            base = base[: -len('_odf_harmonics.txt')]
        else:
            base, _ = os.path.splitext(base)

        parts = [p for p in base.split('_') if p]

        class_name = '_'.join(parts[:-2])
        return class_name.rstrip('_')
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all ODF harmonics files from input directory.
        
        Returns:
            Tuple of (X_all, y_all) where:
                X_all: (n_samples, 18278) array of all samples
                y_all: (n_samples,) array of class labels (integers)
        
        Raises:
            FileNotFoundError: If input directory doesn't exist
            ValueError: If no files found or data format is incorrect
        """
        if not os.path.exists(self.config.INPUT_DIR):
            raise FileNotFoundError(f"Input directory not found: {self.config.INPUT_DIR}")
        
        # Find all .txt files
        file_pattern = os.path.join(self.config.INPUT_DIR, "*.txt")
        all_files = glob.glob(file_pattern)
        
        if len(all_files) == 0:
            raise ValueError(f"No .txt files found in {self.config.INPUT_DIR}")
        
        print(f"Found {len(all_files)} files in {self.config.INPUT_DIR}")
        
        # Organize by class
        class_data_temp = defaultdict(list)
        class_files_temp = defaultdict(list)
        
        for file_path in all_files:
            class_name = self.extract_class_name(file_path)
            
            # Load data (9139 rows × 2 cols: real, imaginary)
            try:
                data = np.loadtxt(file_path)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
            
            # Ensure proper shape
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            
            # Validate shape
            if data.shape != self.config.ODF_SHAPE:
                print(f"Warning: {file_path} has shape {data.shape}, expected {self.config.ODF_SHAPE}. Skipping.")
                continue
            
            # Flatten to 18278-dimensional vector
            flattened = data.flatten()
            
            class_data_temp[class_name].append(flattened)
            class_files_temp[class_name].append(os.path.basename(file_path))
        
        if len(class_data_temp) == 0:
            raise ValueError("No valid data files loaded")
        
        # Convert to arrays
        for class_name in class_data_temp:
            self.class_data[class_name] = np.array(class_data_temp[class_name])
            self.class_files[class_name] = class_files_temp[class_name]
        
        self.class_names_sorted = sorted(self.class_data.keys())
        
        # Pool all data for PCA fitting
        X_all = np.vstack([self.class_data[c] for c in self.class_names_sorted])
        y_all = np.concatenate([
            [i] * len(self.class_data[c]) 
            for i, c in enumerate(self.class_names_sorted)
        ])
        
        return X_all, y_all
    
    def print_summary(self):
        """Print a summary of loaded data."""
        print("\n" + "="*80)
        print("DATA LOADING SUMMARY")
        print("="*80)
        print(f"\n{'Class Name':<40} {'Samples':<10} {'Synthetic Target':<18}")
        print("-" * 60)
        
        for class_name in self.class_names_sorted:
            n_samples = len(self.class_data[class_name])
            target = self.config.target_synthetic_count(n_samples)
            print(f"{class_name:<40} {n_samples:<10} {target:<18}")
        
        total_original = sum(len(self.class_data[c]) for c in self.class_names_sorted)
        total_target = sum(
            self.config.target_synthetic_count(len(self.class_data[c]))
            for c in self.class_names_sorted
        )
        print("-" * 60)
        print(f"{'TOTAL':<40} {total_original:<10} {total_target:<10}")
        print("="*80)
    
    def get_class_data(self, class_name: str) -> np.ndarray:
        """
        Get data for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Array of samples for that class
        """
        return self.class_data.get(class_name, np.array([]))
    
    def get_num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_names_sorted)
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension (18278 for ODF harmonics)."""
        return self.config.ODF_SHAPE[0] * self.config.ODF_SHAPE[1]
