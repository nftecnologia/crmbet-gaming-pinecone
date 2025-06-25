"""
🔍 Advanced Data Quality Module - REAL-TIME MONITORING
Industrial-grade quality automation with anomaly detection and auto-remediation

Author: Agente ETL Industrial - ULTRATHINK  
Created: 2025-06-25
"""

from .realtime_monitor import RealtimeQualityMonitor
from .anomaly_detector import AnomalyDetector
from .schema_evolution import SchemaEvolutionManager
from .data_lineage import DataLineageTracker
from .quality_gates import QualityGateEngine

__all__ = [
    'RealtimeQualityMonitor',
    'AnomalyDetector', 
    'SchemaEvolutionManager',
    'DataLineageTracker',
    'QualityGateEngine'
]