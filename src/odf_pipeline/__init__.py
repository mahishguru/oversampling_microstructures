"""
ODF Harmonics Oversampling Pipeline

A modular pipeline for generating high-quality synthetic samples for 
Orientation Distribution Function (ODF) harmonics data using:
- Latent Monte-Carlo jittering
- Multi-modal augmentation (scaling, mixup, time warping)
- Hybrid ADASYN with contextual adjustment
- Distance-based quality filtering
"""

__version__ = "2.0.0"
__author__ = "ODF Pipeline Team"

from .config import Config
from .data_loader import DataLoader
from .pca_transform import PCATransformer
from .augmentation import Augmentor
from .sampling import HybridSampler
from .filtering import QualityFilter
from .visualization import Visualizer
from .utils import ClassStatistics, FileManager, Logger

# Backward compatibility: DistanceFilter was renamed to QualityFilter in v2.0
DistanceFilter = QualityFilter

__all__ = [
    'Config',
    'DataLoader',
    'PCATransformer',
    'Augmentor',
    'HybridSampler',
    'QualityFilter',
    'Visualizer',
    'ClassStatistics',
    'FileManager',
    'Logger'
]
