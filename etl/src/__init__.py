"""
🏗️ ETL Pipeline CRM Bet ML
Sistema ETL completo com HARDNESS máxima para qualidade de dados

Author: Agente Engenheiro de Dados - ULTRATHINK
Version: 1.0.0
Created: 2025-06-25
"""

__version__ = "1.0.0"
__author__ = "Agente Engenheiro de Dados - ULTRATHINK"
__description__ = "Pipeline ETL robusto para alimentar sistema ML de segmentação de usuários"

# Importações principais
from .etl_pipeline import ETLPipeline, PipelineConfig, PipelineMetrics

# Extractors
from .extractors.datalake_extractor import DataLakeExtractor, ExtractionConfig
from .extractors.transaction_extractor import TransactionExtractor, TransactionExtractionConfig

# Transformers
from .transformers.data_cleaner import DataCleaner, CleaningConfig, CleaningReport
from .transformers.feature_engineer import FeatureEngineer, FeatureConfig, FeatureReport

# Loaders
from .loaders.postgres_loader import PostgresLoader, LoaderConfig, LoadingReport

# Validators
from .validators.data_quality import DataQualityValidator, QualityConfig, QualityReport

__all__ = [
    # Main Pipeline
    'ETLPipeline',
    'PipelineConfig', 
    'PipelineMetrics',
    
    # Extractors
    'DataLakeExtractor',
    'ExtractionConfig',
    'TransactionExtractor',
    'TransactionExtractionConfig',
    
    # Transformers
    'DataCleaner',
    'CleaningConfig',
    'CleaningReport',
    'FeatureEngineer',
    'FeatureConfig',
    'FeatureReport',
    
    # Loaders
    'PostgresLoader',
    'LoaderConfig',
    'LoadingReport',
    
    # Validators
    'DataQualityValidator',
    'QualityConfig',
    'QualityReport'
]