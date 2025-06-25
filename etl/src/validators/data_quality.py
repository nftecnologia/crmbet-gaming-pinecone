"""
🔍 Data Quality Validator - HARDNESS Máxima
Validação rigorosa de qualidade de dados para ML

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import structlog
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import json
import re
from enum import Enum

logger = structlog.get_logger(__name__)

class ValidationSeverity(Enum):
    """Severidade das validações"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ValidationRule:
    """Regra de validação de dados"""
    name: str
    description: str
    severity: ValidationSeverity
    validator_function: Callable
    threshold: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    applies_to_columns: Optional[List[str]] = None

@dataclass
class ValidationResult:
    """Resultado de uma validação"""
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_records: int = 0
    
    @property
    def is_critical(self) -> bool:
        return self.severity == ValidationSeverity.CRITICAL
    
    @property
    def is_blocking(self) -> bool:
        return self.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]

@dataclass
class QualityReport:
    """Relatório completo de qualidade"""
    execution_id: str
    timestamp: datetime
    total_records: int
    validation_results: List[ValidationResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def critical_issues(self) -> List[ValidationResult]:
        return [r for r in self.validation_results if r.is_critical]
    
    @property
    def blocking_issues(self) -> List[ValidationResult]:
        return [r for r in self.validation_results if r.is_blocking]
    
    @property
    def quality_grade(self) -> str:
        if self.overall_score >= 0.95:
            return 'A'
        elif self.overall_score >= 0.85:
            return 'B'
        elif self.overall_score >= 0.70:
            return 'C'
        elif self.overall_score >= 0.50:
            return 'D'
        else:
            return 'F'

@dataclass
class QualityConfig:
    """Configuração de validação de qualidade"""
    
    # Thresholds Críticos (HARDNESS MÁXIMA)
    min_completeness: float = 0.95          # 95% dados completos
    max_outliers: float = 0.05              # Máximo 5% outliers
    max_duplicates: float = 0.02            # Máximo 2% duplicatas
    min_freshness_hours: int = 24           # Dados não podem ter mais de 24h
    
    # Validações de Domínio
    min_user_transactions: int = 1
    max_transaction_value: float = 100000.0
    min_transaction_value: float = 0.01
    
    # Validações Estatísticas
    min_variance_threshold: float = 0.01
    max_skewness: float = 5.0
    max_kurtosis: float = 10.0
    
    # Validações ML-Specific
    min_feature_importance: float = 0.001
    max_correlation_threshold: float = 0.95
    min_samples_per_class: int = 10
    
    # Severidades
    critical_threshold: float = 0.90
    high_threshold: float = 0.75
    medium_threshold: float = 0.50
    
    # Performance
    enable_statistical_tests: bool = True
    enable_ml_validations: bool = True
    enable_business_rules: bool = True

class DataQualityValidator:
    """
    Validador de qualidade de dados com HARDNESS máxima
    Implementa validações rigorosas para garantir dados de alta qualidade para ML
    """
    
    def __init__(self, min_completeness: float = 0.95, 
                 max_outliers: float = 0.05,
                 min_freshness_hours: int = 24,
                 config: Optional[QualityConfig] = None):
        
        self.config = config or QualityConfig()
        self.config.min_completeness = min_completeness
        self.config.max_outliers = max_outliers
        self.config.min_freshness_hours = min_freshness_hours
        
        self.logger = logger.bind(component="DataQualityValidator")
        
        # Inicializa regras de validação
        self._initialize_validation_rules()
        
        # Cache de validações
        self._validation_cache = {}
        
        self.logger.info("DataQualityValidator inicializado", config=self.config.__dict__)
    
    def _initialize_validation_rules(self):
        """Inicializa todas as regras de validação"""
        
        self.validation_rules = [
            # Validações Críticas
            ValidationRule(
                name="completeness_check",
                description="Verifica completude geral dos dados",
                severity=ValidationSeverity.CRITICAL,
                validator_function=self._validate_completeness,
                threshold=self.config.min_completeness
            ),
            
            ValidationRule(
                name="primary_key_integrity",
                description="Verifica integridade da chave primária",
                severity=ValidationSeverity.CRITICAL,
                validator_function=self._validate_primary_key_integrity
            ),
            
            ValidationRule(
                name="data_freshness",
                description="Verifica frescor dos dados",
                severity=ValidationSeverity.HIGH,
                validator_function=self._validate_data_freshness,
                threshold=self.config.min_freshness_hours
            ),
            
            # Validações de Qualidade
            ValidationRule(
                name="duplicate_detection",
                description="Detecta registros duplicados",
                severity=ValidationSeverity.HIGH,
                validator_function=self._validate_duplicates,
                threshold=self.config.max_duplicates
            ),
            
            ValidationRule(
                name="outlier_detection",
                description="Detecta outliers estatísticos",
                severity=ValidationSeverity.MEDIUM,
                validator_function=self._validate_outliers,
                threshold=self.config.max_outliers
            ),
            
            ValidationRule(
                name="data_types_validation",
                description="Valida tipos de dados",
                severity=ValidationSeverity.HIGH,
                validator_function=self._validate_data_types
            ),
            
            # Validações de Domínio
            ValidationRule(
                name="business_rules_validation",
                description="Valida regras de negócio específicas",
                severity=ValidationSeverity.HIGH,
                validator_function=self._validate_business_rules
            ),
            
            ValidationRule(
                name="value_range_validation",
                description="Valida faixas de valores",
                severity=ValidationSeverity.MEDIUM,
                validator_function=self._validate_value_ranges
            ),
            
            # Validações Estatísticas
            ValidationRule(
                name="statistical_distribution",
                description="Analisa distribuições estatísticas",
                severity=ValidationSeverity.LOW,
                validator_function=self._validate_statistical_distribution
            ),
            
            ValidationRule(
                name="feature_correlation",
                description="Detecta correlações excessivas entre features",
                severity=ValidationSeverity.MEDIUM,
                validator_function=self._validate_feature_correlation
            ),
            
            # Validações ML-Específicas
            ValidationRule(
                name="ml_readiness",
                description="Verifica preparação para ML",
                severity=ValidationSeverity.HIGH,
                validator_function=self._validate_ml_readiness
            )
        ]
    
    def validate_raw_data(self, df: pd.DataFrame, execution_id: Optional[str] = None) -> QualityReport:
        """
        Valida dados brutos (pré-processamento)
        
        Args:
            df: DataFrame com dados brutos
            execution_id: ID da execução
            
        Returns:
            Relatório de qualidade
        """
        return self._execute_validation_suite(
            df, 
            execution_id or f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            validation_type="raw"
        )
    
    def validate_transformed_data(self, df: pd.DataFrame, execution_id: Optional[str] = None) -> QualityReport:
        """
        Valida dados transformados (pós-processamento)
        
        Args:
            df: DataFrame com dados transformados
            execution_id: ID da execução
            
        Returns:
            Relatório de qualidade
        """
        return self._execute_validation_suite(
            df,
            execution_id or f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            validation_type="transformed"
        )
    
    def _execute_validation_suite(self, df: pd.DataFrame, execution_id: str, 
                                validation_type: str) -> QualityReport:
        """Executa suite completa de validações"""
        
        self.logger.info(
            "Iniciando validação de qualidade",
            execution_id=execution_id,
            type=validation_type,
            records=len(df)
        )
        
        # Inicializa relatório
        report = QualityReport(
            execution_id=execution_id,
            timestamp=datetime.now(),
            total_records=len(df)
        )
        
        try:
            # Executa todas as validações
            for rule in self.validation_rules:
                # Pula validações ML para dados brutos
                if validation_type == "raw" and rule.name in ["ml_readiness", "feature_correlation"]:
                    continue
                
                try:
                    result = rule.validator_function(df, rule)
                    report.validation_results.append(result)
                    
                    # Log resultado
                    self.logger.info(
                        "Validação executada",
                        rule=rule.name,
                        passed=result.passed,
                        score=result.score,
                        severity=result.severity.value
                    )
                    
                except Exception as e:
                    error_msg = f"Erro na validação {rule.name}: {str(e)}"
                    report.errors.append(error_msg)
                    self.logger.error("Erro na validação", rule=rule.name, error=str(e))
            
            # Calcula score geral e determina resultado
            report.overall_score = self._calculate_overall_score(report)
            report.passed = self._determine_validation_result(report)
            
            # Gera recomendações
            report.recommendations = self._generate_recommendations(report)
            
            self.logger.info(
                "Validação de qualidade concluída",
                execution_id=execution_id,
                overall_score=report.overall_score,
                grade=report.quality_grade,
                passed=report.passed,
                critical_issues=len(report.critical_issues)
            )
            
            return report
            
        except Exception as e:
            self.logger.error("Erro na suite de validação", error=str(e))
            raise
    
    def _validate_completeness(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Valida completude dos dados"""
        
        if df.empty:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message="DataFrame está vazio"
            )
        
        # Calcula completude geral
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        completeness = (total_cells - null_cells) / total_cells
        
        # Completude por coluna
        column_completeness = {}
        critical_columns = ['user_id', 'amount', 'created_at']
        
        for column in df.columns:
            col_completeness = 1 - (df[column].isnull().sum() / len(df))
            column_completeness[column] = col_completeness
            
            # Verifica colunas críticas
            if column in critical_columns and col_completeness < 1.0:
                return ValidationResult(
                    rule_name=rule.name,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    score=col_completeness,
                    message=f"Coluna crítica {column} tem valores nulos",
                    details={'column_completeness': column_completeness},
                    affected_records=df[column].isnull().sum()
                )
        
        passed = completeness >= rule.threshold
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=completeness,
            message=f"Completude geral: {completeness:.3f}",
            details={'column_completeness': column_completeness}
        )
    
    def _validate_primary_key_integrity(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Valida integridade da chave primária"""
        
        primary_key_column = 'user_id'
        
        if primary_key_column not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message=f"Chave primária {primary_key_column} não encontrada"
            )
        
        # Verifica valores nulos
        null_count = df[primary_key_column].isnull().sum()
        if null_count > 0:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message=f"Chave primária tem {null_count} valores nulos",
                affected_records=null_count
            )
        
        # Verifica valores vazios
        empty_count = (df[primary_key_column].astype(str).str.strip() == '').sum()
        if empty_count > 0:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message=f"Chave primária tem {empty_count} valores vazios",
                affected_records=empty_count
            )
        
        # Verifica duplicatas
        duplicate_count = df[primary_key_column].duplicated().sum()
        unique_ratio = 1 - (duplicate_count / len(df))
        
        passed = duplicate_count == 0
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=unique_ratio,
            message=f"Integridade PK: {duplicate_count} duplicatas encontradas",
            details={'duplicate_count': duplicate_count, 'unique_values': df[primary_key_column].nunique()},
            affected_records=duplicate_count
        )
    
    def _validate_data_freshness(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Valida frescor dos dados"""
        
        timestamp_columns = ['created_at', 'updated_at', 'last_activity', 'transaction_date']
        found_timestamp_column = None
        
        for col in timestamp_columns:
            if col in df.columns:
                found_timestamp_column = col
                break
        
        if not found_timestamp_column:
            return ValidationResult(
                rule_name=rule.name,
                severity=ValidationSeverity.LOW,
                passed=True,
                score=0.5,
                message="Nenhuma coluna de timestamp encontrada para validar frescor"
            )
        
        # Converte para datetime se necessário
        try:
            timestamp_series = pd.to_datetime(df[found_timestamp_column], errors='coerce')
        except Exception:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message=f"Erro ao converter {found_timestamp_column} para datetime"
            )
        
        # Calcula idade dos dados
        current_time = datetime.now()
        max_age_threshold = timedelta(hours=rule.threshold)
        
        oldest_record = timestamp_series.min()
        newest_record = timestamp_series.max()
        
        if pd.isna(oldest_record) or pd.isna(newest_record):
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message="Timestamps inválidos encontrados"
            )
        
        # Verifica se há dados muito antigos
        age_of_newest = current_time - newest_record.to_pydatetime()
        age_of_oldest = current_time - oldest_record.to_pydatetime()
        
        # Score baseado na idade dos dados mais recentes
        freshness_score = max(0.0, 1.0 - (age_of_newest.total_seconds() / (max_age_threshold.total_seconds() * 2)))
        
        passed = age_of_newest <= max_age_threshold
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=freshness_score,
            message=f"Dados mais recentes têm {age_of_newest.total_seconds()/3600:.1f}h",
            details={
                'oldest_record': oldest_record.isoformat(),
                'newest_record': newest_record.isoformat(),
                'age_hours': age_of_newest.total_seconds() / 3600
            }
        )
    
    def _validate_duplicates(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Detecta registros duplicados"""
        
        if df.empty:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                score=1.0,
                message="DataFrame vazio - sem duplicatas"
            )
        
        # Duplicatas exatas
        exact_duplicates = df.duplicated().sum()
        
        # Duplicatas baseadas em chaves de negócio
        business_key_columns = ['user_id']
        if 'created_at' in df.columns:
            business_key_columns.append('created_at')
        
        business_duplicates = 0
        if all(col in df.columns for col in business_key_columns):
            business_duplicates = df.duplicated(subset=business_key_columns).sum()
        
        total_duplicates = max(exact_duplicates, business_duplicates)
        duplicate_ratio = total_duplicates / len(df)
        
        passed = duplicate_ratio <= rule.threshold
        score = 1.0 - duplicate_ratio
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Encontradas {total_duplicates} duplicatas ({duplicate_ratio:.3f}%)",
            details={
                'exact_duplicates': exact_duplicates,
                'business_duplicates': business_duplicates,
                'duplicate_ratio': duplicate_ratio
            },
            affected_records=total_duplicates
        )
    
    def _validate_outliers(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Detecta outliers estatísticos"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                score=1.0,
                message="Nenhuma coluna numérica para detectar outliers"
            )
        
        total_outliers = 0
        column_outliers = {}
        
        for column in numeric_columns:
            if column in ['user_id']:  # Skip ID columns
                continue
            
            # Método IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            column_outliers[column] = outliers
            total_outliers += outliers
        
        outlier_ratio = total_outliers / (len(df) * len(numeric_columns))
        passed = outlier_ratio <= rule.threshold
        score = 1.0 - min(1.0, outlier_ratio / rule.threshold)
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Detectados {total_outliers} outliers ({outlier_ratio:.3f}%)",
            details={'column_outliers': column_outliers},
            affected_records=total_outliers
        )
    
    def _validate_data_types(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Valida tipos de dados"""
        
        expected_types = {
            'user_id': 'object',
            'amount': ['float64', 'int64'],
            'created_at': 'datetime64[ns]'
        }
        
        type_violations = []
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                
                if isinstance(expected_type, list):
                    if actual_type not in expected_type:
                        type_violations.append(f"{column}: esperado {expected_type}, encontrado {actual_type}")
                else:
                    if actual_type != expected_type:
                        type_violations.append(f"{column}: esperado {expected_type}, encontrado {actual_type}")
        
        passed = len(type_violations) == 0
        score = 1.0 - (len(type_violations) / len(expected_types))
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Violações de tipo: {len(type_violations)}",
            details={'violations': type_violations}
        )
    
    def _validate_business_rules(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Valida regras de negócio específicas"""
        
        violations = []
        
        # Regra 1: Valores de transação devem estar dentro dos limites
        if 'amount' in df.columns:
            invalid_amounts = df[
                (df['amount'] < self.config.min_transaction_value) | 
                (df['amount'] > self.config.max_transaction_value)
            ]
            
            if len(invalid_amounts) > 0:
                violations.append(f"Valores de transação inválidos: {len(invalid_amounts)} registros")
        
        # Regra 2: Usuários devem ter atividade mínima
        if 'user_id' in df.columns:
            user_transaction_counts = df['user_id'].value_counts()
            inactive_users = user_transaction_counts[
                user_transaction_counts < self.config.min_user_transactions
            ]
            
            if len(inactive_users) > 0:
                violations.append(f"Usuários com atividade insuficiente: {len(inactive_users)}")
        
        # Regra 3: Features categóricas devem ter valores válidos
        categorical_rules = {
            'game_type': ['crash', 'cassino', 'esportes', 'poker', 'slots', 'unknown'],
            'transaction_type': ['deposit', 'withdrawal', 'bet', 'win', 'bonus', 'refund'],
            'status': ['completed', 'pending', 'failed', 'cancelled']
        }
        
        for column, valid_values in categorical_rules.items():
            if column in df.columns:
                invalid_values = df[~df[column].isin(valid_values + ['unknown', None])]
                if len(invalid_values) > 0:
                    violations.append(f"Valores inválidos em {column}: {len(invalid_values)} registros")
        
        passed = len(violations) == 0
        score = 1.0 - min(1.0, len(violations) / 10)  # Normaliza por número máximo de regras
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Violações de regras de negócio: {len(violations)}",
            details={'violations': violations}
        )
    
    def _validate_value_ranges(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Valida faixas de valores"""
        
        value_range_rules = {
            'age': (0, 120),
            'session_duration': (0, 86400),  # máximo 24h em segundos
            'bet_amount': (0, 100000),
            'balance': (-100000, 1000000)
        }
        
        violations = []
        
        for column, (min_val, max_val) in value_range_rules.items():
            if column in df.columns:
                out_of_range = df[
                    (df[column] < min_val) | (df[column] > max_val)
                ]
                
                if len(out_of_range) > 0:
                    violations.append(
                        f"{column}: {len(out_of_range)} valores fora da faixa [{min_val}, {max_val}]"
                    )
        
        passed = len(violations) == 0
        score = 1.0 - min(1.0, len(violations) / len(value_range_rules))
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Violações de faixa de valores: {len(violations)}",
            details={'violations': violations}
        )
    
    def _validate_statistical_distribution(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Analisa distribuições estatísticas"""
        
        if not self.config.enable_statistical_tests:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                score=1.0,
                message="Testes estatísticos desabilitados"
            )
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                score=1.0,
                message="Nenhuma coluna numérica para análise estatística"
            )
        
        statistical_issues = []
        
        for column in numeric_columns:
            if column in ['user_id']:  # Skip ID columns
                continue
            
            try:
                # Calcula estatísticas
                skewness = stats.skew(df[column].dropna())
                kurtosis = stats.kurtosis(df[column].dropna())
                variance = df[column].var()
                
                # Verifica problemas
                if abs(skewness) > self.config.max_skewness:
                    statistical_issues.append(f"{column}: skewness extrema ({skewness:.2f})")
                
                if abs(kurtosis) > self.config.max_kurtosis:
                    statistical_issues.append(f"{column}: kurtosis extrema ({kurtosis:.2f})")
                
                if variance < self.config.min_variance_threshold:
                    statistical_issues.append(f"{column}: variância muito baixa ({variance:.6f})")
                    
            except Exception as e:
                statistical_issues.append(f"{column}: erro no cálculo estatístico")
        
        passed = len(statistical_issues) == 0
        score = 1.0 - min(1.0, len(statistical_issues) / len(numeric_columns))
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Problemas estatísticos: {len(statistical_issues)}",
            details={'issues': statistical_issues}
        )
    
    def _validate_feature_correlation(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Detecta correlações excessivas entre features"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                score=1.0,
                message="Poucas colunas numéricas para análise de correlação"
            )
        
        try:
            # Calcula matriz de correlação
            correlation_matrix = df[numeric_columns].corr()
            
            # Encontra correlações altas (excluindo diagonal)
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    
                    if corr_value > self.config.max_correlation_threshold:
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        high_correlations.append(f"{col1} <-> {col2}: {corr_value:.3f}")
            
            passed = len(high_correlations) == 0
            score = 1.0 - min(1.0, len(high_correlations) / 10)
            
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=passed,
                score=score,
                message=f"Correlações altas detectadas: {len(high_correlations)}",
                details={'high_correlations': high_correlations}
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                score=0.0,
                message=f"Erro na análise de correlação: {str(e)}"
            )
    
    def _validate_ml_readiness(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Verifica preparação para ML"""
        
        if not self.config.enable_ml_validations:
            return ValidationResult(
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                score=1.0,
                message="Validações ML desabilitadas"
            )
        
        ml_issues = []
        
        # 1. Verifica se há features suficientes
        feature_columns = [col for col in df.columns if col not in ['user_id', 'created_at', 'updated_at']]
        
        if len(feature_columns) < 5:
            ml_issues.append(f"Poucas features para ML: {len(feature_columns)}")
        
        # 2. Verifica balanceamento de classes categóricas
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column not in ['user_id']:
                value_counts = df[column].value_counts()
                min_class_size = value_counts.min()
                
                if min_class_size < self.config.min_samples_per_class:
                    ml_issues.append(f"{column}: classe com poucos samples ({min_class_size})")
        
        # 3. Verifica se há variabilidade suficiente
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column not in ['user_id']:
                unique_ratio = df[column].nunique() / len(df)
                
                if unique_ratio < 0.01:  # Menos de 1% valores únicos
                    ml_issues.append(f"{column}: baixa variabilidade ({unique_ratio:.3f})")
        
        # 4. Verifica escala das features
        for column in numeric_columns:
            if column not in ['user_id']:
                col_range = df[column].max() - df[column].min()
                col_std = df[column].std()
                
                if col_std > 0 and col_range / col_std > 1000:  # Range muito grande
                    ml_issues.append(f"{column}: escala inadequada para ML")
        
        passed = len(ml_issues) == 0
        score = 1.0 - min(1.0, len(ml_issues) / 10)
        
        return ValidationResult(
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            score=score,
            message=f"Problemas de preparação ML: {len(ml_issues)}",
            details={'ml_issues': ml_issues}
        )
    
    def _calculate_overall_score(self, report: QualityReport) -> float:
        """Calcula score geral de qualidade"""
        
        if not report.validation_results:
            return 0.0
        
        # Pesos por severidade
        severity_weights = {
            ValidationSeverity.CRITICAL: 3.0,
            ValidationSeverity.HIGH: 2.0,
            ValidationSeverity.MEDIUM: 1.5,
            ValidationSeverity.LOW: 1.0,
            ValidationSeverity.INFO: 0.5
        }
        
        weighted_scores = []
        
        for result in report.validation_results:
            weight = severity_weights.get(result.severity, 1.0)
            weighted_scores.append(result.score * weight)
        
        if weighted_scores:
            return sum(weighted_scores) / len(weighted_scores)
        else:
            return 0.0
    
    def _determine_validation_result(self, report: QualityReport) -> bool:
        """Determina se validação passou"""
        
        # Falha se há issues críticos
        if report.critical_issues:
            return False
        
        # Falha se score geral é muito baixo
        if report.overall_score < self.config.critical_threshold:
            return False
        
        # Falha se muitos issues bloqueantes
        if len(report.blocking_issues) > 3:
            return False
        
        return True
    
    def _generate_recommendations(self, report: QualityReport) -> List[str]:
        """Gera recomendações baseadas nos resultados"""
        
        recommendations = []
        
        # Recomendações baseadas em issues críticos
        for result in report.critical_issues:
            if "completeness" in result.rule_name:
                recommendations.append("Revisar processo de coleta para reduzir dados faltantes")
            elif "primary_key" in result.rule_name:
                recommendations.append("Implementar validação de chave primária na origem")
        
        # Recomendações baseadas em score
        if report.overall_score < 0.7:
            recommendations.append("Qualidade geral baixa - revisar pipeline completo")
        elif report.overall_score < 0.85:
            recommendations.append("Implementar melhorias incrementais de qualidade")
        
        # Recomendações específicas
        outlier_results = [r for r in report.validation_results if "outlier" in r.rule_name]
        if outlier_results and not outlier_results[0].passed:
            recommendations.append("Implementar detecção de outliers mais rigorosa")
        
        duplicate_results = [r for r in report.validation_results if "duplicate" in r.rule_name]
        if duplicate_results and not duplicate_results[0].passed:
            recommendations.append("Melhorar lógica de deduplicação")
        
        return recommendations

# Função utilitária para teste
def test_data_quality_validator():
    """Teste do validador de qualidade"""
    
    # Cria dados de teste com problemas intencionais
    test_data = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3', 'user1', None],  # Duplicata e nulo
        'amount': [100.0, -50.0, 200.0, 50000.0, 75.0],        # Negativo e outlier
        'game_type': ['crash', 'invalid_game', 'esportes', 'crash', 'cassino'],
        'created_at': [
            datetime.now(),
            datetime.now() - timedelta(days=30),  # Dado antigo
            datetime.now() - timedelta(hours=1),
            datetime.now(),
            None  # Data nula
        ]
    })
    
    print("Dados de teste:")
    print(test_data)
    print(f"Shape: {test_data.shape}")
    
    # Inicializa validador
    validator = DataQualityValidator()
    
    # Executa validação em dados brutos
    raw_report = validator.validate_raw_data(test_data)
    
    print(f"\nRelatório de Qualidade - Dados Brutos:")
    print(f"Score Geral: {raw_report.overall_score:.3f} (Nota: {raw_report.quality_grade})")
    print(f"Passou: {raw_report.passed}")
    print(f"Issues Críticos: {len(raw_report.critical_issues)}")
    print(f"Issues Bloqueantes: {len(raw_report.blocking_issues)}")
    
    if raw_report.errors:
        print(f"Erros: {raw_report.errors}")
    
    if raw_report.recommendations:
        print(f"Recomendações: {raw_report.recommendations}")
    
    print("\nDetalhes por Validação:")
    for result in raw_report.validation_results:
        status = "✅ PASSOU" if result.passed else "❌ FALHOU"
        print(f"{status} {result.rule_name}: {result.message} (Score: {result.score:.3f})")

if __name__ == "__main__":
    test_data_quality_validator()