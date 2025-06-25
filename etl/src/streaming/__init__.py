"""
⚡ Streaming ETL Module - REAL-TIME TB+/HOUR PROCESSING
Near real-time data processing with <5min latency for industrial scale

Author: Agente ETL Industrial - ULTRATHINK  
Created: 2025-06-25
"""

from .kafka_consumer import KafkaStreamConsumer
from .stream_processor import StreamProcessor
from .windowing_manager import WindowingManager
from .realtime_quality import RealtimeQualityMonitor

__all__ = [
    'KafkaStreamConsumer',
    'StreamProcessor', 
    'WindowingManager',
    'RealtimeQualityMonitor'
]