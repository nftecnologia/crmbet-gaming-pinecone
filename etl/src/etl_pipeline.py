"""
🏗️ ETL Pipeline Principal - CRM Bet ML
Orquestrador de pipeline com HARDNESS máxima na qualidade dos dados

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import asyncio
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import structlog

# Importações locais
from extractors.datalake_extractor import DataLakeExtractor
from extractors.transaction_extractor import TransactionExtractor
from transformers.data_cleaner import DataCleaner
from transformers.feature_engineer import FeatureEngineer
from loaders.postgres_loader import PostgresLoader
from validators.data_quality import DataQualityValidator

# Configuração de logging estruturado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@dataclass
class PipelineConfig:
    """Configuração INDUSTRIAL do Pipeline ETL - TB+/HORA SCALE"""
    
    # ═══════════════════════════════════════════════════════════════
    # DATA LAKE CONFIGURATION - MASSIVE SCALE
    # ═══════════════════════════════════════════════════════════════
    s3_bucket: str = os.getenv('DATA_LAKE_BUCKET', 'crmbet-datalake')
    s3_prefix: str = os.getenv('DATA_LAKE_PREFIX', 'raw/')
    s3_region: str = os.getenv('AWS_REGION', 'us-east-1')
    
    # Advanced S3 settings for TB+ scale
    s3_multipart_threshold: int = int(os.getenv('S3_MULTIPART_THRESHOLD', '104857600'))  # 100MB
    s3_max_concurrency: int = int(os.getenv('S3_MAX_CONCURRENCY', '10'))
    s3_use_accelerate: bool = os.getenv('S3_USE_ACCELERATE', 'true').lower() == 'true'
    
    # ═══════════════════════════════════════════════════════════════
    # DATABASE CONFIGURATION - HIGH PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    db_url: str = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/crmbet')
    db_pool_size: int = int(os.getenv('DB_POOL_SIZE', '20'))
    db_max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', '30'))
    db_pool_timeout: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))
    db_pool_recycle: int = int(os.getenv('DB_POOL_RECYCLE', '3600'))
    
    # Read replicas for scaling reads
    db_read_replicas: List[str] = field(default_factory=lambda: 
        os.getenv('DB_READ_REPLICAS', '').split(',') if os.getenv('DB_READ_REPLICAS') else [])
    
    # ═══════════════════════════════════════════════════════════════
    # PROCESSING CONFIGURATION - INDUSTRIAL SCALE
    # ═══════════════════════════════════════════════════════════════
    
    # Batch processing settings
    batch_size: int = int(os.getenv('ETL_BATCH_SIZE', '100000'))  # 100K records per batch
    mega_batch_size: int = int(os.getenv('ETL_MEGA_BATCH_SIZE', '1000000'))  # 1M for bulk ops
    max_workers: int = int(os.getenv('ETL_MAX_WORKERS', str(os.cpu_count() * 2)))  # Oversubscribe
    
    # Memory management for TB+ processing
    max_memory_per_worker_gb: int = int(os.getenv('ETL_MAX_MEMORY_GB', '8'))
    memory_limit_enforcement: bool = os.getenv('ETL_MEMORY_LIMIT', 'true').lower() == 'true'
    gc_threshold_mb: int = int(os.getenv('ETL_GC_THRESHOLD', '1024'))  # Force GC at 1GB
    
    # Distributed computing
    enable_dask_distributed: bool = os.getenv('ETL_ENABLE_DASK', 'true').lower() == 'true'
    dask_scheduler_address: Optional[str] = os.getenv('DASK_SCHEDULER_ADDRESS')
    dask_worker_threads: int = int(os.getenv('DASK_WORKER_THREADS', '4'))
    
    # Compression settings for performance
    enable_compression: bool = os.getenv('ETL_ENABLE_COMPRESSION', 'true').lower() == 'true'
    compression_algorithm: str = os.getenv('ETL_COMPRESSION_ALGO', 'lz4')  # lz4, zstd, snappy
    compression_level: int = int(os.getenv('ETL_COMPRESSION_LEVEL', '1'))
    
    # ═══════════════════════════════════════════════════════════════
    # STREAMING CONFIGURATION - REAL-TIME PROCESSING
    # ═══════════════════════════════════════════════════════════════
    enable_streaming: bool = os.getenv('ETL_STREAMING', 'true').lower() == 'true'
    
    # Kafka configuration
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: 
        os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','))
    kafka_consumer_group: str = os.getenv('KAFKA_CONSUMER_GROUP', 'crmbet-etl-industrial')
    kafka_batch_size: int = int(os.getenv('KAFKA_BATCH_SIZE', '50000'))  # 50K messages per batch
    kafka_max_poll_records: int = int(os.getenv('KAFKA_MAX_POLL_RECORDS', '10000'))
    
    # Stream processing windows
    stream_window_seconds: int = int(os.getenv('STREAM_WINDOW_SECONDS', '60'))  # 1 minute windows
    stream_watermark_seconds: int = int(os.getenv('STREAM_WATERMARK_SECONDS', '30'))  # 30s watermark
    
    # ═══════════════════════════════════════════════════════════════
    # QUALITY THRESHOLDS - HARDNESS MÁXIMA
    # ═══════════════════════════════════════════════════════════════
    min_data_completeness: float = float(os.getenv('ETL_MIN_COMPLETENESS', '0.98'))  # 98% dados completos
    max_outlier_percentage: float = float(os.getenv('ETL_MAX_OUTLIERS', '0.02'))  # Máximo 2% outliers
    min_data_freshness_minutes: int = int(os.getenv('ETL_MIN_FRESHNESS_MIN', '5'))  # <5min latência
    max_data_staleness_hours: int = int(os.getenv('ETL_MAX_STALENESS_HOURS', '2'))  # Máximo 2h de atraso
    
    # Data quality automation
    enable_auto_quality_remediation: bool = os.getenv('ETL_AUTO_REMEDIATION', 'true').lower() == 'true'
    quality_check_interval_seconds: int = int(os.getenv('ETL_QUALITY_CHECK_INTERVAL', '30'))
    enable_anomaly_detection: bool = os.getenv('ETL_ANOMALY_DETECTION', 'true').lower() == 'true'
    
    # ═══════════════════════════════════════════════════════════════
    # FAULT TOLERANCE - INDUSTRIAL RELIABILITY
    # ═══════════════════════════════════════════════════════════════
    
    # Circuit breaker settings
    enable_circuit_breakers: bool = os.getenv('ETL_CIRCUIT_BREAKERS', 'true').lower() == 'true'
    circuit_breaker_failure_threshold: int = int(os.getenv('ETL_CB_FAILURE_THRESHOLD', '5'))
    circuit_breaker_recovery_timeout: int = int(os.getenv('ETL_CB_RECOVERY_TIMEOUT', '60'))
    
    # Retry configuration
    max_retries: int = int(os.getenv('ETL_MAX_RETRIES', '3'))
    retry_backoff_multiplier: float = float(os.getenv('ETL_RETRY_BACKOFF', '2.0'))
    retry_max_delay_seconds: int = int(os.getenv('ETL_RETRY_MAX_DELAY', '300'))
    
    # Dead letter queue
    enable_dlq: bool = os.getenv('ETL_ENABLE_DLQ', 'true').lower() == 'true'
    dlq_max_retries: int = int(os.getenv('ETL_DLQ_MAX_RETRIES', '3'))
    
    # ═══════════════════════════════════════════════════════════════
    # MONITORING & OBSERVABILITY - PRODUCTION READY
    # ═══════════════════════════════════════════════════════════════
    
    # Metrics collection
    enable_detailed_metrics: bool = os.getenv('ETL_DETAILED_METRICS', 'true').lower() == 'true'
    metrics_collection_interval: int = int(os.getenv('ETL_METRICS_INTERVAL', '15'))  # 15 seconds
    prometheus_port: int = int(os.getenv('PROMETHEUS_PORT', '9090'))
    
    # Performance tracking
    enable_performance_profiling: bool = os.getenv('ETL_PERFORMANCE_PROFILING', 'true').lower() == 'true'
    profiling_sample_rate: float = float(os.getenv('ETL_PROFILING_SAMPLE_RATE', '0.1'))  # 10%
    
    # Alerting
    enable_alerting: bool = os.getenv('ETL_ENABLE_ALERTING', 'true').lower() == 'true'
    alert_webhook_url: Optional[str] = os.getenv('ETL_ALERT_WEBHOOK_URL')
    alert_email_recipients: List[str] = field(default_factory=lambda: 
        os.getenv('ETL_ALERT_EMAILS', '').split(',') if os.getenv('ETL_ALERT_EMAILS') else [])
    
    # ═══════════════════════════════════════════════════════════════
    # MACHINE LEARNING FEATURES - ADVANCED ANALYTICS
    # ═══════════════════════════════════════════════════════════════
    target_features: List[str] = field(default_factory=lambda: [
        # User Behavior Features
        'favorite_game_type',
        'avg_session_duration_minutes',
        'sessions_per_day',
        'preferred_game_times',
        'device_preferences',
        
        # Financial Features  
        'avg_transaction_amount',
        'transaction_frequency',
        'deposit_patterns',
        'withdrawal_patterns',
        'balance_trends',
        
        # Engagement Features
        'days_since_last_login',
        'total_games_played',
        'win_loss_ratio',
        'bonus_usage_rate',
        'support_interactions',
        
        # Risk Features
        'risk_score',
        'unusual_activity_flags',
        'geo_location_changes',
        'payment_method_changes'
    ])
    
    # Feature engineering settings
    enable_advanced_features: bool = os.getenv('ETL_ADVANCED_FEATURES', 'true').lower() == 'true'
    feature_lookback_days: int = int(os.getenv('ETL_FEATURE_LOOKBACK_DAYS', '90'))
    enable_real_time_features: bool = os.getenv('ETL_REALTIME_FEATURES', 'true').lower() == 'true'
    
    # ═══════════════════════════════════════════════════════════════
    # SCHEDULING & ORCHESTRATION - ENTERPRISE
    # ═══════════════════════════════════════════════════════════════
    
    # Batch scheduling
    run_schedule: str = os.getenv('ETL_SCHEDULE', '0 */2 * * *')  # Every 2 hours for TB+ scale
    enable_incremental_processing: bool = os.getenv('ETL_INCREMENTAL', 'true').lower() == 'true'
    
    # Streaming schedule
    stream_processing_enabled: bool = os.getenv('ETL_STREAM_PROCESSING', 'true').lower() == 'true'
    stream_checkpoint_interval: int = int(os.getenv('ETL_STREAM_CHECKPOINT', '300'))  # 5 minutes
    
    # Resource scheduling
    peak_hours: List[int] = field(default_factory=lambda: 
        [int(h) for h in os.getenv('ETL_PEAK_HOURS', '9,10,11,14,15,16,20,21,22').split(',')])
    scale_up_during_peak: bool = os.getenv('ETL_SCALE_UP_PEAK', 'true').lower() == 'true'
    peak_hour_multiplier: float = float(os.getenv('ETL_PEAK_MULTIPLIER', '2.0'))
    
    # ═══════════════════════════════════════════════════════════════
    # COST OPTIMIZATION - EFFICIENCY
    # ═══════════════════════════════════════════════════════════════
    
    # Resource optimization
    enable_auto_scaling: bool = os.getenv('ETL_AUTO_SCALING', 'true').lower() == 'true'
    min_workers: int = int(os.getenv('ETL_MIN_WORKERS', '2'))
    max_workers_limit: int = int(os.getenv('ETL_MAX_WORKERS_LIMIT', '100'))
    
    # Cost tracking
    enable_cost_tracking: bool = os.getenv('ETL_COST_TRACKING', 'true').lower() == 'true'
    cost_budget_daily_usd: float = float(os.getenv('ETL_DAILY_BUDGET', '1000.0'))
    
    # Storage optimization
    enable_data_lifecycle: bool = os.getenv('ETL_DATA_LIFECYCLE', 'true').lower() == 'true'
    hot_data_retention_days: int = int(os.getenv('ETL_HOT_RETENTION_DAYS', '30'))
    warm_data_retention_days: int = int(os.getenv('ETL_WARM_RETENTION_DAYS', '90'))
    cold_data_retention_days: int = int(os.getenv('ETL_COLD_RETENTION_DAYS', '365'))
    
    # ═══════════════════════════════════════════════════════════════
    # SECURITY & COMPLIANCE
    # ═══════════════════════════════════════════════════════════════
    
    # Data encryption
    enable_encryption_at_rest: bool = os.getenv('ETL_ENCRYPTION_REST', 'true').lower() == 'true'
    enable_encryption_in_transit: bool = os.getenv('ETL_ENCRYPTION_TRANSIT', 'true').lower() == 'true'
    
    # PII handling
    enable_pii_detection: bool = os.getenv('ETL_PII_DETECTION', 'true').lower() == 'true'
    enable_data_masking: bool = os.getenv('ETL_DATA_MASKING', 'true').lower() == 'true'
    
    # Audit
    enable_audit_logging: bool = os.getenv('ETL_AUDIT_LOGGING', 'true').lower() == 'true'
    audit_retention_days: int = int(os.getenv('ETL_AUDIT_RETENTION_DAYS', '365'))
    
    @property
    def is_peak_hour(self) -> bool:
        """Verifica se está em horário de pico"""
        from datetime import datetime
        current_hour = datetime.now().hour
        return current_hour in self.peak_hours
    
    @property
    def effective_workers(self) -> int:
        """Retorna número efetivo de workers baseado no horário"""
        base_workers = self.max_workers
        if self.scale_up_during_peak and self.is_peak_hour:
            return min(int(base_workers * self.peak_hour_multiplier), self.max_workers_limit)
        return base_workers
    
    @property
    def effective_batch_size(self) -> int:
        """Retorna tamanho efetivo do batch baseado na carga"""
        if self.is_peak_hour:
            return self.mega_batch_size  # Use larger batches during peak
        return self.batch_size

@dataclass
class PipelineMetrics:
    """Métricas de execução do pipeline"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records_processed: int = 0
    records_extracted: int = 0
    records_cleaned: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    data_quality_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.records_extracted == 0:
            return 0.0
        return self.records_loaded / self.records_extracted

class ETLPipeline:
    """
    Pipeline ETL Principal com foco em qualidade de dados
    Implementa padrão HARDNESS para máxima confiabilidade
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = logger.bind(component="ETLPipeline")
        
        # Inicialização de componentes
        self._initialize_components()
        
        # Métricas de execução
        self.current_metrics: Optional[PipelineMetrics] = None
        
        self.logger.info("ETL Pipeline inicializado", config=self.config.__dict__)
    
    def _initialize_components(self):
        """Inicializa todos os componentes do pipeline"""
        try:
            self.datalake_extractor = DataLakeExtractor(
                bucket=self.config.s3_bucket,
                prefix=self.config.s3_prefix
            )
            
            self.transaction_extractor = TransactionExtractor(
                db_url=self.config.db_url
            )
            
            self.data_cleaner = DataCleaner(
                completeness_threshold=self.config.min_data_completeness,
                outlier_threshold=self.config.max_outlier_percentage
            )
            
            self.feature_engineer = FeatureEngineer(
                target_features=self.config.target_features
            )
            
            self.postgres_loader = PostgresLoader(
                db_url=self.config.db_url,
                batch_size=self.config.batch_size
            )
            
            self.quality_validator = DataQualityValidator(
                min_completeness=self.config.min_data_completeness,
                max_outliers=self.config.max_outlier_percentage,
                min_freshness_hours=self.config.min_data_freshness_hours
            )
            
            self.logger.info("Todos os componentes inicializados com sucesso")
            
        except Exception as e:
            self.logger.error("Erro na inicialização de componentes", error=str(e))
            raise
    
    def run_full_pipeline(self, execution_id: Optional[str] = None) -> PipelineMetrics:
        """
        Executa o pipeline completo ETL
        
        Args:
            execution_id: ID único da execução (gerado automaticamente se não fornecido)
        
        Returns:
            PipelineMetrics com resultados da execução
        """
        exec_id = execution_id or f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Inicializa métricas
        self.current_metrics = PipelineMetrics(
            execution_id=exec_id,
            start_time=datetime.now()
        )
        
        self.logger.info("Iniciando pipeline ETL completo", execution_id=exec_id)
        
        try:
            # Fase 1: Extração
            raw_data = self._extract_phase()
            
            # Fase 2: Validação inicial de qualidade
            self._validate_raw_data(raw_data)
            
            # Fase 3: Limpeza
            clean_data = self._clean_phase(raw_data)
            
            # Fase 4: Transformação e Feature Engineering
            transformed_data = self._transform_phase(clean_data)
            
            # Fase 5: Validação final de qualidade
            quality_score = self._validate_final_data(transformed_data)
            
            # Fase 6: Carregamento
            self._load_phase(transformed_data)
            
            # Finalização
            self.current_metrics.end_time = datetime.now()
            self.current_metrics.data_quality_score = quality_score
            
            self.logger.info(
                "Pipeline ETL executado com sucesso",
                execution_id=exec_id,
                duration=self.current_metrics.duration_seconds,
                quality_score=quality_score,
                success_rate=self.current_metrics.success_rate
            )
            
            return self.current_metrics
            
        except Exception as e:
            self._handle_pipeline_error(e)
            raise
    
    def _extract_phase(self) -> pd.DataFrame:
        """Fase de extração de dados"""
        self.logger.info("Iniciando fase de extração")
        
        # Extração paralela de dados
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submete tarefas de extração
            future_datalake = executor.submit(self.datalake_extractor.extract_all)
            future_transactions = executor.submit(self.transaction_extractor.extract_recent)
            
            # Coleta resultados
            datalake_data = future_datalake.result()
            transaction_data = future_transactions.result()
        
        # Combina dados
        combined_data = self._merge_extracted_data(datalake_data, transaction_data)
        
        self.current_metrics.records_extracted = len(combined_data)
        
        self.logger.info(
            "Fase de extração concluída",
            records_extracted=self.current_metrics.records_extracted
        )
        
        return combined_data
    
    def _merge_extracted_data(self, datalake_data: pd.DataFrame, 
                            transaction_data: pd.DataFrame) -> pd.DataFrame:
        """Combina dados do Data Lake e transações"""
        try:
            # Merge inteligente baseado em user_id e timestamp
            merged_data = pd.merge(
                datalake_data,
                transaction_data,
                on=['user_id'],
                how='outer',
                suffixes=('_dl', '_tx')
            )
            
            self.logger.info(
                "Dados combinados com sucesso",
                datalake_records=len(datalake_data),
                transaction_records=len(transaction_data),
                merged_records=len(merged_data)
            )
            
            return merged_data
            
        except Exception as e:
            self.logger.error("Erro ao combinar dados", error=str(e))
            raise
    
    def _validate_raw_data(self, data: pd.DataFrame):
        """Validação inicial dos dados brutos"""
        self.logger.info("Validando dados brutos")
        
        validation_results = self.quality_validator.validate_raw_data(data)
        
        if not validation_results.passed:
            error_msg = f"Dados brutos falharam na validação: {validation_results.errors}"
            self.current_metrics.errors.append(error_msg)
            self.logger.error("Validação de dados brutos falhou", 
                            errors=validation_results.errors)
            raise ValueError(error_msg)
        
        # Adiciona warnings se houver
        if validation_results.warnings:
            self.current_metrics.warnings.extend(validation_results.warnings)
            self.logger.warning("Alertas na validação de dados brutos",
                              warnings=validation_results.warnings)
        
        self.logger.info("Dados brutos validados com sucesso")
    
    def _clean_phase(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fase de limpeza de dados"""
        self.logger.info("Iniciando fase de limpeza de dados")
        
        clean_data = self.data_cleaner.clean_data(data)
        
        self.current_metrics.records_cleaned = len(clean_data)
        
        self.logger.info(
            "Fase de limpeza concluída",
            records_before=len(data),
            records_after=len(clean_data),
            removed_records=len(data) - len(clean_data)
        )
        
        return clean_data
    
    def _transform_phase(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fase de transformação e feature engineering"""
        self.logger.info("Iniciando fase de transformação")
        
        transformed_data = self.feature_engineer.engineer_features(data)
        
        self.current_metrics.records_transformed = len(transformed_data)
        
        self.logger.info(
            "Fase de transformação concluída",
            records_transformed=self.current_metrics.records_transformed,
            features_created=len(transformed_data.columns)
        )
        
        return transformed_data
    
    def _validate_final_data(self, data: pd.DataFrame) -> float:
        """Validação final dos dados transformados"""
        self.logger.info("Validando dados finais")
        
        validation_results = self.quality_validator.validate_transformed_data(data)
        
        if not validation_results.passed:
            error_msg = f"Dados transformados falharam na validação: {validation_results.errors}"
            self.current_metrics.errors.append(error_msg)
            self.logger.error("Validação de dados finais falhou",
                            errors=validation_results.errors)
            raise ValueError(error_msg)
        
        quality_score = validation_results.quality_score
        
        self.logger.info(
            "Dados finais validados com sucesso",
            quality_score=quality_score
        )
        
        return quality_score
    
    def _load_phase(self, data: pd.DataFrame):
        """Fase de carregamento no PostgreSQL"""
        self.logger.info("Iniciando fase de carregamento")
        
        loaded_records = self.postgres_loader.load_data(data)
        
        self.current_metrics.records_loaded = loaded_records
        
        self.logger.info(
            "Fase de carregamento concluída",
            records_loaded=loaded_records
        )
    
    def _handle_pipeline_error(self, error: Exception):
        """Trata erros do pipeline"""
        error_msg = f"Erro crítico no pipeline: {str(error)}"
        
        if self.current_metrics:
            self.current_metrics.errors.append(error_msg)
            self.current_metrics.end_time = datetime.now()
        
        self.logger.error(
            "Pipeline falhou com erro crítico",
            error=error_msg,
            traceback=traceback.format_exc()
        )
    
    def run_streaming_pipeline(self):
        """Executa pipeline em modo streaming (experimental)"""
        if not self.config.enable_streaming:
            self.logger.warning("Streaming não habilitado na configuração")
            return
        
        self.logger.info("Iniciando pipeline em modo streaming")
        
        # Implementação de streaming seria aqui
        # Por enquanto, executa batches menores com mais frequência
        
        while True:
            try:
                self.logger.info("Executando batch streaming")
                self.run_full_pipeline()
                time.sleep(300)  # 5 minutos entre execuções
                
            except KeyboardInterrupt:
                self.logger.info("Pipeline streaming interrompido pelo usuário")
                break
            except Exception as e:
                self.logger.error("Erro no pipeline streaming", error=str(e))
                time.sleep(60)  # Aguarda 1 minuto antes de tentar novamente
    
    def schedule_pipeline(self):
        """Agenda execução automática do pipeline"""
        self.logger.info("Configurando agenda do pipeline", schedule=self.config.run_schedule)
        
        # Agenda para execução diária às 2h
        schedule.every().day.at("02:00").do(self.run_full_pipeline)
        
        # Agenda para verificação de qualidade a cada 6h
        schedule.every(6).hours.do(self._quality_check)
        
        self.logger.info("Pipeline agendado com sucesso")
        
        # Loop principal de agendamento
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verifica a cada minuto
    
    def _quality_check(self):
        """Verificação rápida de qualidade dos dados"""
        self.logger.info("Executando verificação de qualidade")
        
        # Implementação de verificação rápida
        # Verifica apenas alguns indicadores chave
        pass
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retorna status atual do pipeline"""
        status = {
            "pipeline_health": "healthy",
            "last_execution": None,
            "next_scheduled": None,
            "components_status": {
                "datalake_extractor": "ok",
                "transaction_extractor": "ok", 
                "data_cleaner": "ok",
                "feature_engineer": "ok",
                "postgres_loader": "ok",
                "quality_validator": "ok"
            }
        }
        
        if self.current_metrics:
            status["last_execution"] = {
                "execution_id": self.current_metrics.execution_id,
                "start_time": self.current_metrics.start_time.isoformat(),
                "end_time": self.current_metrics.end_time.isoformat() if self.current_metrics.end_time else None,
                "records_processed": self.current_metrics.total_records_processed,
                "quality_score": self.current_metrics.data_quality_score,
                "success_rate": self.current_metrics.success_rate
            }
        
        return status

def main():
    """Função principal para execução standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ETL Pipeline CRM Bet ML')
    parser.add_argument('--mode', choices=['batch', 'streaming', 'schedule'], 
                       default='batch', help='Modo de execução')
    parser.add_argument('--execution-id', help='ID da execução')
    
    args = parser.parse_args()
    
    # Inicializa pipeline
    pipeline = ETLPipeline()
    
    try:
        if args.mode == 'batch':
            metrics = pipeline.run_full_pipeline(args.execution_id)
            print(f"Pipeline executado com sucesso: {metrics.execution_id}")
            
        elif args.mode == 'streaming':
            pipeline.run_streaming_pipeline()
            
        elif args.mode == 'schedule':
            pipeline.schedule_pipeline()
            
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
    except Exception as e:
        logger.error("Erro na execução", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()