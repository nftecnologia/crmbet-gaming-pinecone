"""
🏗️ Distributed Processing Module - TB+/HOUR SCALE
Industrial-grade distributed computing for massive data processing

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

from .dask_manager import DaskManager
from .distributed_extractor import DistributedExtractor
from .parallel_processor import ParallelProcessor
from .cluster_coordinator import ClusterCoordinator

__all__ = [
    'DaskManager',
    'DistributedExtractor', 
    'ParallelProcessor',
    'ClusterCoordinator'
]