"""
🚨 Real-time Quality Monitor - CONTINUOUS DATA QUALITY  
Advanced monitoring with ML-powered anomaly detection and auto-remediation

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import structlog

# Data processing imports
import pandas as pd
import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis
import psutil

# Async imports
import aiofiles
import httpx

logger = structlog.get_logger(__name__)

# Prometheus metrics
QUALITY_SCORE = Gauge('data_quality_score', 'Data quality score', ['dataset', 'dimension'])
ANOMALIES_DETECTED = Counter('anomalies_detected_total', 'Total anomalies detected', ['type', 'severity'])
QUALITY_CHECKS_PERFORMED = Counter('quality_checks_performed_total', 'Total quality checks', ['check_type'])
DATA_FRESHNESS = Gauge('data_freshness_seconds', 'Data freshness in seconds', ['source'])
SCHEMA_VIOLATIONS = Counter('schema_violations_total', 'Schema violations', ['field', 'violation_type'])

class QualityDimension(Enum):
    """Dimensões de qualidade de dados"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy" 
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"

class AlertSeverity(Enum):
    """Severidade dos alertas"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class QualityThresholds:
    """Thresholds configuráveis para qualidade"""
    
    # Completeness thresholds
    min_completeness: float = 0.95
    critical_completeness: float = 0.85
    
    # Accuracy thresholds  
    max_error_rate: float = 0.05
    critical_error_rate: float = 0.15
    
    # Consistency thresholds
    max_inconsistency_rate: float = 0.02
    critical_inconsistency_rate: float = 0.10
    
    # Freshness thresholds (em segundos)
    max_staleness_seconds: int = 3600  # 1 hora
    critical_staleness_seconds: int = 7200  # 2 horas
    
    # Volume thresholds
    min_volume_threshold: float = 0.8  # 80% do volume esperado
    max_volume_threshold: float = 2.0  # 200% do volume esperado
    
    # Outlier detection
    outlier_z_threshold: float = 3.0
    outlier_iqr_factor: float = 1.5

@dataclass
class QualityMetrics:
    """Métricas de qualidade calculadas"""
    timestamp: datetime
    dataset: str
    
    # Scores por dimensão (0-1)
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    validity_score: float = 0.0
    timeliness_score: float = 0.0
    uniqueness_score: float = 0.0
    
    # Score geral
    overall_score: float = 0.0
    
    # Contadores
    total_records: int = 0
    null_records: int = 0
    duplicate_records: int = 0
    invalid_records: int = 0
    
    # Anomalias detectadas
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def grade(self) -> str:
        """Nota de qualidade baseada no score"""
        if self.overall_score >= 0.95: return 'A'
        elif self.overall_score >= 0.85: return 'B'
        elif self.overall_score >= 0.70: return 'C'
        elif self.overall_score >= 0.50: return 'D'
        else: return 'F'

@dataclass
class QualityAlert:
    """Alerta de qualidade"""
    timestamp: datetime
    dataset: str
    dimension: QualityDimension
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'dataset': self.dataset,
            'dimension': self.dimension.value,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'details': self.details
        }

class RealtimeQualityMonitor:
    """
    Monitor de qualidade em tempo real com capacidades avançadas:
    - Monitoramento contínuo de múltiplas dimensões
    - ML-powered anomaly detection
    - Auto-remediation para problemas conhecidos
    - Alertas inteligentes com deduplicação
    - Tracking de qualidade histórica
    """
    
    def __init__(self, 
                 dataset_name: str,
                 thresholds: Optional[QualityThresholds] = None,
                 redis_client: Optional[redis.Redis] = None):
        
        self.dataset_name = dataset_name
        self.thresholds = thresholds or QualityThresholds()
        self.logger = logger.bind(component="RealtimeQualityMonitor", dataset=dataset_name)
        
        # Redis para cache e estado distribuído
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Estado interno
        self.running = False
        self.quality_history = deque(maxlen=1000)  # Últimas 1000 medições
        self.alert_history = deque(maxlen=500)     # Últimos 500 alertas
        self.baseline_models = {}                  # Modelos ML para anomaly detection
        
        # Buffers para processamento em batch
        self.metrics_buffer = deque(maxlen=100)
        self.anomaly_buffer = deque(maxlen=50)
        
        # Estatísticas adaptativas
        self.adaptive_thresholds = dict(self.thresholds.__dict__)
        self.baseline_stats = {}
        
        # Alert deduplication
        self.recent_alerts = defaultdict(lambda: defaultdict(datetime))
        self.alert_cooldown_minutes = 15
        
        # Background tasks
        self.monitoring_task = None
        self.baseline_update_task = None
        
        self.logger.info("Monitor de qualidade inicializado", thresholds=self.thresholds.__dict__)
    
    async def start_monitoring(self):
        """Inicia monitoramento em tempo real"""
        
        try:
            self.logger.info("Iniciando monitoramento de qualidade em tempo real")
            self.running = True
            
            # Carrega baseline existente
            await self._load_baseline_models()
            
            # Inicia tasks em background
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.baseline_update_task = asyncio.create_task(self._baseline_update_loop())
            
            self.logger.info("Monitoramento de qualidade iniciado")
            
        except Exception as e:
            self.logger.error("Erro ao iniciar monitoramento", error=str(e))
            raise
    
    async def stop_monitoring(self):
        """Para monitoramento"""
        
        try:
            self.logger.info("Parando monitoramento de qualidade")
            self.running = False
            
            # Cancela tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.baseline_update_task:
                self.baseline_update_task.cancel()
            
            # Salva baseline atual
            await self._save_baseline_models()
            
            self.logger.info("Monitoramento de qualidade parado")
            
        except Exception as e:
            self.logger.error("Erro ao parar monitoramento", error=str(e))
    
    async def analyze_data_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """
        Analisa qualidade de dados de forma abrangente
        
        Args:
            df: DataFrame para analisar
            
        Returns:
            Métricas de qualidade calculadas
        """
        
        start_time = time.time()
        
        try:
            metrics = QualityMetrics(
                timestamp=datetime.now(),
                dataset=self.dataset_name,
                total_records=len(df)
            )
            
            # Análises paralelas para performance
            tasks = [
                self._analyze_completeness(df, metrics),
                self._analyze_accuracy(df, metrics),
                self._analyze_consistency(df, metrics),
                self._analyze_validity(df, metrics),
                self._analyze_timeliness(df, metrics),
                self._analyze_uniqueness(df, metrics),
            ]
            
            await asyncio.gather(*tasks)
            
            # Calcula score geral (média ponderada)
            weights = {
                'completeness': 0.25,
                'accuracy': 0.20,
                'consistency': 0.15,
                'validity': 0.15,
                'timeliness': 0.15,
                'uniqueness': 0.10
            }
            
            metrics.overall_score = (
                metrics.completeness_score * weights['completeness'] +
                metrics.accuracy_score * weights['accuracy'] +
                metrics.consistency_score * weights['consistency'] +
                metrics.validity_score * weights['validity'] +
                metrics.timeliness_score * weights['timeliness'] +
                metrics.uniqueness_score * weights['uniqueness']
            )
            
            # Detecção de anomalias ML
            anomalies = await self._detect_ml_anomalies(df, metrics)
            metrics.anomalies.extend(anomalies)
            
            # Atualiza histórico
            self.quality_history.append(metrics)
            
            # Atualiza métricas Prometheus
            self._update_prometheus_metrics(metrics)
            
            # Gera alertas se necessário
            alerts = await self._generate_alerts(metrics)
            
            # Salva no Redis para consulta distribuída
            await self._cache_metrics(metrics)
            
            processing_time = time.time() - start_time
            self.logger.info(
                "Análise de qualidade concluída",
                overall_score=metrics.overall_score,
                grade=metrics.grade,
                anomalies=len(metrics.anomalies),
                alerts=len(alerts),
                processing_time_ms=processing_time * 1000
            )
            
            # Increment counter
            QUALITY_CHECKS_PERFORMED.labels(check_type='full_analysis').inc()
            
            return metrics
            
        except Exception as e:
            self.logger.error("Erro na análise de qualidade", error=str(e))
            raise
    
    async def _analyze_completeness(self, df: pd.DataFrame, metrics: QualityMetrics):
        """Analisa completeness dos dados"""
        
        if df.empty:
            metrics.completeness_score = 0.0
            return
        
        # Calcula valores nulos
        null_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        total_nulls = null_counts.sum()
        
        metrics.null_records = total_nulls
        metrics.completeness_score = 1.0 - (total_nulls / total_cells)
        
        # Análise por coluna crítica
        critical_columns = ['user_id', 'amount', 'timestamp']
        for col in critical_columns:
            if col in df.columns:
                col_completeness = 1.0 - (df[col].isnull().sum() / len(df))
                if col_completeness < self.thresholds.critical_completeness:
                    metrics.anomalies.append({
                        'type': 'critical_column_incomplete',
                        'column': col,
                        'completeness': col_completeness,
                        'severity': 'critical'
                    })
    
    async def _analyze_accuracy(self, df: pd.DataFrame, metrics: QualityMetrics):
        """Analisa accuracy usando regras de negócio"""
        
        if df.empty:
            metrics.accuracy_score = 1.0
            return
        
        invalid_count = 0
        total_checks = 0
        
        # Validações específicas por tipo de dados
        if 'amount' in df.columns:
            # Valores negativos onde não deveria haver
            invalid_amounts = df[df['amount'] < 0]
            invalid_count += len(invalid_amounts)
            total_checks += len(df)
        
        if 'email' in df.columns:
            # Emails inválidos (regex básico)
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = df[~df['email'].str.match(email_pattern, na=False)]
            invalid_count += len(invalid_emails)
            total_checks += len(df.dropna(subset=['email']))
        
        if 'age' in df.columns:
            # Idades impossíveis
            invalid_ages = df[(df['age'] < 0) | (df['age'] > 150)]
            invalid_count += len(invalid_ages)
            total_checks += len(df.dropna(subset=['age']))
        
        # Score de accuracy
        if total_checks > 0:
            metrics.accuracy_score = 1.0 - (invalid_count / total_checks)
            metrics.invalid_records = invalid_count
        else:
            metrics.accuracy_score = 1.0
    
    async def _analyze_consistency(self, df: pd.DataFrame, metrics: QualityMetrics):
        """Analisa consistency entre campos relacionados"""
        
        if df.empty:
            metrics.consistency_score = 1.0
            return
        
        inconsistency_count = 0
        total_checks = 0
        
        # Exemplo: consistência entre deposit_amount e transaction_type
        if all(col in df.columns for col in ['amount', 'transaction_type']):
            # Deposits devem ter valor positivo
            deposits = df[df['transaction_type'] == 'deposit']
            if len(deposits) > 0:
                negative_deposits = deposits[deposits['amount'] <= 0]
                inconsistency_count += len(negative_deposits)
                total_checks += len(deposits)
            
            # Withdrawals devem ter valor positivo (valor absoluto)
            withdrawals = df[df['transaction_type'] == 'withdrawal']
            if len(withdrawals) > 0:
                invalid_withdrawals = withdrawals[withdrawals['amount'] <= 0]
                inconsistency_count += len(invalid_withdrawals)
                total_checks += len(withdrawals)
        
        # Score de consistency
        if total_checks > 0:
            metrics.consistency_score = 1.0 - (inconsistency_count / total_checks)
        else:
            metrics.consistency_score = 1.0
    
    async def _analyze_validity(self, df: pd.DataFrame, metrics: QualityMetrics):
        """Analisa validity usando schemas e constraints"""
        
        if df.empty:
            metrics.validity_score = 1.0
            return
        
        validity_violations = 0
        total_validations = 0
        
        # Validações de tipo de dados
        for column in df.columns:
            if column in ['user_id']:
                # IDs não devem ser vazios
                empty_ids = df[df[column].astype(str).str.strip() == '']
                validity_violations += len(empty_ids)
                total_validations += len(df)
            
            elif 'date' in column.lower() or 'time' in column.lower():
                # Datas devem ser válidas
                try:
                    pd.to_datetime(df[column], errors='coerce')
                    invalid_dates = df[pd.to_datetime(df[column], errors='coerce').isna()]
                    validity_violations += len(invalid_dates)
                    total_validations += len(df.dropna(subset=[column]))
                except:
                    pass
        
        # Score de validity
        if total_validations > 0:
            metrics.validity_score = 1.0 - (validity_violations / total_validations)
        else:
            metrics.validity_score = 1.0
    
    async def _analyze_timeliness(self, df: pd.DataFrame, metrics: QualityMetrics):
        """Analisa timeliness dos dados"""
        
        if df.empty:
            metrics.timeliness_score = 1.0
            return
        
        # Busca coluna de timestamp
        timestamp_cols = [col for col in df.columns if 
                         'time' in col.lower() or 'date' in col.lower()]
        
        if not timestamp_cols:
            metrics.timeliness_score = 0.5  # Score neutro se não há timestamp
            return
        
        timestamp_col = timestamp_cols[0]  # Usa primeira coluna encontrada
        
        try:
            # Converte para datetime
            timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            # Calcula idade dos dados
            current_time = datetime.now()
            staleness_seconds = (current_time - timestamps.max()).total_seconds()
            
            # Score baseado na staleness
            if staleness_seconds <= self.thresholds.max_staleness_seconds:
                metrics.timeliness_score = 1.0
            elif staleness_seconds <= self.thresholds.critical_staleness_seconds:
                # Score linear entre max e critical
                ratio = (staleness_seconds - self.thresholds.max_staleness_seconds) / \
                       (self.thresholds.critical_staleness_seconds - self.thresholds.max_staleness_seconds)
                metrics.timeliness_score = 1.0 - ratio
            else:
                metrics.timeliness_score = 0.0
            
            # Atualiza métrica Prometheus
            DATA_FRESHNESS.labels(source=self.dataset_name).set(staleness_seconds)
            
        except Exception as e:
            self.logger.warning("Erro na análise de timeliness", error=str(e))
            metrics.timeliness_score = 0.5
    
    async def _analyze_uniqueness(self, df: pd.DataFrame, metrics: QualityMetrics):
        """Analisa uniqueness (duplicatas)"""
        
        if df.empty:
            metrics.uniqueness_score = 1.0
            return
        
        # Detecta duplicatas exatas
        duplicate_count = df.duplicated().sum()
        
        # Detecta duplicatas de chave de negócio
        if 'user_id' in df.columns:
            business_key_duplicates = df.duplicated(subset=['user_id']).sum()
            duplicate_count = max(duplicate_count, business_key_duplicates)
        
        metrics.duplicate_records = duplicate_count
        metrics.uniqueness_score = 1.0 - (duplicate_count / len(df))
    
    async def _detect_ml_anomalies(self, df: pd.DataFrame, metrics: QualityMetrics) -> List[Dict[str, Any]]:
        """Detecta anomalias usando ML"""
        
        anomalies = []
        
        try:
            # Seleciona apenas colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return anomalies
            
            numeric_data = df[numeric_cols].dropna()
            
            if len(numeric_data) < 10:  # Precisa de dados mínimos
                return anomalies
            
            # Normaliza dados
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Isolation Forest para detecção de outliers
            isolation_forest = IsolationForest(
                contamination=0.1,  # Espera 10% de outliers
                random_state=42,
                n_jobs=-1
            )
            
            outlier_predictions = isolation_forest.fit_predict(scaled_data)
            outlier_scores = isolation_forest.score_samples(scaled_data)
            
            # Identifica outliers
            outlier_indices = np.where(outlier_predictions == -1)[0]
            
            for idx in outlier_indices:
                anomalies.append({
                    'type': 'statistical_outlier',
                    'index': int(idx),
                    'anomaly_score': float(outlier_scores[idx]),
                    'affected_columns': list(numeric_cols),
                    'severity': 'medium' if outlier_scores[idx] < -0.5 else 'low'
                })
            
            # Volume anomaly detection
            current_volume = len(df)
            if self.baseline_stats.get('avg_volume'):
                expected_volume = self.baseline_stats['avg_volume']
                volume_ratio = current_volume / expected_volume
                
                if (volume_ratio < self.thresholds.min_volume_threshold or 
                    volume_ratio > self.thresholds.max_volume_threshold):
                    
                    anomalies.append({
                        'type': 'volume_anomaly',
                        'current_volume': current_volume,
                        'expected_volume': expected_volume,
                        'ratio': volume_ratio,
                        'severity': 'high' if volume_ratio < 0.5 or volume_ratio > 3.0 else 'medium'
                    })
            
            # Schema drift detection
            current_schema = set(df.columns)
            if self.baseline_stats.get('expected_schema'):
                expected_schema = set(self.baseline_stats['expected_schema'])
                
                missing_cols = expected_schema - current_schema
                extra_cols = current_schema - expected_schema
                
                if missing_cols or extra_cols:
                    anomalies.append({
                        'type': 'schema_drift',
                        'missing_columns': list(missing_cols),
                        'extra_columns': list(extra_cols),
                        'severity': 'critical' if missing_cols else 'medium'
                    })
            
            # Update Prometheus metrics
            for anomaly in anomalies:
                ANOMALIES_DETECTED.labels(
                    type=anomaly['type'], 
                    severity=anomaly['severity']
                ).inc()
            
        except Exception as e:
            self.logger.error("Erro na detecção de anomalias ML", error=str(e))
        
        return anomalies
    
    async def _generate_alerts(self, metrics: QualityMetrics) -> List[QualityAlert]:
        """Gera alertas baseados nas métricas"""
        
        alerts = []
        current_time = datetime.now()
        
        # Alert para cada dimensão de qualidade
        quality_checks = [
            (QualityDimension.COMPLETENESS, metrics.completeness_score, self.thresholds.min_completeness),
            (QualityDimension.ACCURACY, metrics.accuracy_score, 1.0 - self.thresholds.max_error_rate),
            (QualityDimension.CONSISTENCY, metrics.consistency_score, 1.0 - self.thresholds.max_inconsistency_rate),
            (QualityDimension.TIMELINESS, metrics.timeliness_score, 0.8),  # 80% threshold
            (QualityDimension.UNIQUENESS, metrics.uniqueness_score, 0.95)  # 95% threshold
        ]
        
        for dimension, score, threshold in quality_checks:
            if score < threshold:
                # Determina severidade
                if score < threshold * 0.7:  # Muito baixo
                    severity = AlertSeverity.CRITICAL
                elif score < threshold * 0.85:  # Baixo
                    severity = AlertSeverity.HIGH
                else:  # Marginalmente baixo
                    severity = AlertSeverity.MEDIUM
                
                # Verifica cooldown de alertas
                last_alert_time = self.recent_alerts[dimension.value][severity.value]
                if (current_time - last_alert_time).seconds < self.alert_cooldown_minutes * 60:
                    continue  # Skip se muito recente
                
                alert = QualityAlert(
                    timestamp=current_time,
                    dataset=self.dataset_name,
                    dimension=dimension,
                    severity=severity,
                    message=f"{dimension.value.title()} score ({score:.3f}) below threshold ({threshold:.3f})",
                    current_value=score,
                    threshold_value=threshold
                )
                
                alerts.append(alert)
                self.recent_alerts[dimension.value][severity.value] = current_time
        
        # Alerts para anomalias críticas
        for anomaly in metrics.anomalies:
            if anomaly.get('severity') in ['critical', 'high']:
                alert = QualityAlert(
                    timestamp=current_time,
                    dataset=self.dataset_name,
                    dimension=QualityDimension.ACCURACY,  # Default dimension
                    severity=AlertSeverity.CRITICAL if anomaly['severity'] == 'critical' else AlertSeverity.HIGH,
                    message=f"Anomaly detected: {anomaly['type']}",
                    current_value=0.0,
                    threshold_value=1.0,
                    details=anomaly
                )
                alerts.append(alert)
        
        # Salva alertas
        for alert in alerts:
            self.alert_history.append(alert)
            await self._send_alert(alert)
        
        return alerts
    
    async def _send_alert(self, alert: QualityAlert):
        """Envia alerta via múltiplos canais"""
        
        try:
            # Log estruturado
            self.logger.warning(
                "Quality Alert",
                dataset=alert.dataset,
                dimension=alert.dimension.value,
                severity=alert.severity.value,
                message=alert.message,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value
            )
            
            # Salva no Redis para dashboards
            alert_key = f"quality_alert:{self.dataset_name}:{alert.timestamp.isoformat()}"
            await self._redis_set(alert_key, json.dumps(alert.to_dict()), ex=86400)  # 24h TTL
            
            # Webhook notification (se configurado)
            webhook_url = os.getenv('QUALITY_WEBHOOK_URL')
            if webhook_url:
                async with httpx.AsyncClient() as client:
                    await client.post(webhook_url, json=alert.to_dict(), timeout=5.0)
            
        except Exception as e:
            self.logger.error("Erro ao enviar alerta", error=str(e))
    
    async def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        
        while self.running:
            try:
                # Processa buffer de métricas se houver
                if self.metrics_buffer:
                    await self._process_metrics_buffer()
                
                # Atualiza estatísticas adaptativas
                await self._update_adaptive_statistics()
                
                # Cleanup de dados antigos
                await self._cleanup_old_data()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Erro no loop de monitoramento", error=str(e))
                await asyncio.sleep(60)
    
    async def _baseline_update_loop(self):
        """Loop para atualização de baseline ML"""
        
        while self.running:
            try:
                # Atualiza baseline a cada 6 horas
                await asyncio.sleep(6 * 3600)
                
                if len(self.quality_history) > 50:  # Precisa de histórico mínimo
                    await self._update_baseline_models()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Erro na atualização de baseline", error=str(e))
    
    async def _load_baseline_models(self):
        """Carrega modelos baseline do Redis"""
        
        try:
            baseline_key = f"quality_baseline:{self.dataset_name}"
            baseline_data = await self._redis_get(baseline_key)
            
            if baseline_data:
                self.baseline_stats = json.loads(baseline_data)
                self.logger.info("Baseline carregado", stats_keys=list(self.baseline_stats.keys()))
            
        except Exception as e:
            self.logger.warning("Erro ao carregar baseline", error=str(e))
    
    async def _save_baseline_models(self):
        """Salva modelos baseline no Redis"""
        
        try:
            baseline_key = f"quality_baseline:{self.dataset_name}"
            await self._redis_set(baseline_key, json.dumps(self.baseline_stats), ex=7*24*3600)  # 7 dias
            
        except Exception as e:
            self.logger.error("Erro ao salvar baseline", error=str(e))
    
    async def _update_baseline_models(self):
        """Atualiza modelos baseline com dados recentes"""
        
        try:
            if not self.quality_history:
                return
            
            # Calcula estatísticas do histórico
            recent_metrics = list(self.quality_history)[-100:]  # Últimas 100 medições
            
            volumes = [m.total_records for m in recent_metrics]
            overall_scores = [m.overall_score for m in recent_metrics]
            
            self.baseline_stats.update({
                'avg_volume': statistics.mean(volumes),
                'std_volume': statistics.stdev(volumes) if len(volumes) > 1 else 0,
                'avg_quality_score': statistics.mean(overall_scores),
                'std_quality_score': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                'last_updated': datetime.now().isoformat()
            })
            
            # Salva baseline atualizado
            await self._save_baseline_models()
            
            self.logger.info("Baseline atualizado", stats=self.baseline_stats)
            
        except Exception as e:
            self.logger.error("Erro ao atualizar baseline", error=str(e))
    
    def _update_prometheus_metrics(self, metrics: QualityMetrics):
        """Atualiza métricas Prometheus"""
        
        # Scores por dimensão
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='overall').set(metrics.overall_score)
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='completeness').set(metrics.completeness_score)
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='accuracy').set(metrics.accuracy_score)
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='consistency').set(metrics.consistency_score)
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='validity').set(metrics.validity_score)
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='timeliness').set(metrics.timeliness_score)
        QUALITY_SCORE.labels(dataset=self.dataset_name, dimension='uniqueness').set(metrics.uniqueness_score)
    
    async def _cache_metrics(self, metrics: QualityMetrics):
        """Cache métricas no Redis"""
        
        try:
            # Cache métricas atuais
            current_key = f"quality_metrics:{self.dataset_name}:current"
            metrics_dict = {
                'timestamp': metrics.timestamp.isoformat(),
                'overall_score': metrics.overall_score,
                'grade': metrics.grade,
                'total_records': metrics.total_records,
                'anomalies_count': len(metrics.anomalies)
            }
            
            await self._redis_set(current_key, json.dumps(metrics_dict), ex=3600)  # 1h TTL
            
            # Cache série temporal (últimas 24h)
            timeseries_key = f"quality_timeseries:{self.dataset_name}"
            await self._redis_zadd(timeseries_key, {
                json.dumps(metrics_dict): metrics.timestamp.timestamp()
            })
            
            # Remove dados antigos (>24h)
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            await self._redis_zremrangebyscore(timeseries_key, 0, cutoff_time)
            
        except Exception as e:
            self.logger.error("Erro ao fazer cache das métricas", error=str(e))
    
    async def _redis_get(self, key: str) -> Optional[str]:
        """Get assíncrono do Redis"""
        try:
            return self.redis_client.get(key)
        except:
            return None
    
    async def _redis_set(self, key: str, value: str, ex: Optional[int] = None):
        """Set assíncrono do Redis"""
        try:
            self.redis_client.set(key, value, ex=ex)
        except:
            pass
    
    async def _redis_zadd(self, key: str, mapping: Dict):
        """ZAdd assíncrono do Redis"""
        try:
            self.redis_client.zadd(key, mapping)
        except:
            pass
    
    async def _redis_zremrangebyscore(self, key: str, min_score: float, max_score: float):
        """ZRemRangeByScore assíncrono do Redis"""
        try:
            self.redis_client.zremrangebyscore(key, min_score, max_score)
        except:
            pass
    
    async def get_quality_status(self) -> Dict[str, Any]:
        """Retorna status atual de qualidade"""
        
        if not self.quality_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.quality_history[-1]
        
        return {
            'dataset': self.dataset_name,
            'last_check': latest_metrics.timestamp.isoformat(),
            'overall_score': latest_metrics.overall_score,
            'grade': latest_metrics.grade,
            'dimensions': {
                'completeness': latest_metrics.completeness_score,
                'accuracy': latest_metrics.accuracy_score,
                'consistency': latest_metrics.consistency_score,
                'validity': latest_metrics.validity_score,
                'timeliness': latest_metrics.timeliness_score,
                'uniqueness': latest_metrics.uniqueness_score
            },
            'anomalies': len(latest_metrics.anomalies),
            'recent_alerts': len([a for a in self.alert_history 
                                if (datetime.now() - a.timestamp).seconds < 3600]),
            'baseline_stats': self.baseline_stats,
            'running': self.running
        }

# Context manager para uso fácil
class QualityMonitorContext:
    """Context manager para monitor de qualidade"""
    
    def __init__(self, dataset_name: str, thresholds: Optional[QualityThresholds] = None):
        self.monitor = RealtimeQualityMonitor(dataset_name, thresholds)
    
    async def __aenter__(self):
        await self.monitor.start_monitoring()
        return self.monitor
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.monitor.stop_monitoring()