"""
🧹 Data Cleaner - Limpeza com HARDNESS Máxima
Limpeza robusta e validação rigorosa de dados para ML

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import re
import logging
import structlog
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
import json

logger = structlog.get_logger(__name__)

@dataclass
class CleaningConfig:
    """Configuração de limpeza de dados"""
    # Thresholds de Qualidade (HARDNESS MÁXIMA)
    min_completeness_threshold: float = 0.95  # 95% dados completos
    max_outlier_percentage: float = 0.05       # Máximo 5% outliers
    max_duplicate_percentage: float = 0.02     # Máximo 2% duplicatas
    
    # Validações de Dados
    min_user_activity_days: int = 1
    max_transaction_value: float = 100000.0
    min_transaction_value: float = 0.01
    
    # String Cleaning
    normalize_strings: bool = True
    remove_special_chars: bool = True
    standardize_currencies: bool = True
    
    # Outlier Detection
    outlier_methods: List[str] = field(default_factory=lambda: [
        'iqr', 'zscore', 'isolation_forest', 'dbscan'
    ])
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    
    # Date/Time Validation
    min_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    max_date: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))
    
    # Feature Specific Rules
    email_validation: bool = True
    phone_validation: bool = True
    currency_standardization: bool = True
    
    # Performance
    chunk_size: int = 50000
    enable_parallel_processing: bool = True

@dataclass
class CleaningReport:
    """Relatório de limpeza de dados"""
    original_records: int = 0
    cleaned_records: int = 0
    removed_duplicates: int = 0
    removed_outliers: int = 0
    fixed_nulls: int = 0
    standardized_fields: int = 0
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    
    @property
    def retention_rate(self) -> float:
        if self.original_records == 0:
            return 0.0
        return self.cleaned_records / self.original_records
    
    @property
    def cleaning_summary(self) -> Dict[str, Any]:
        return {
            'original_records': self.original_records,
            'cleaned_records': self.cleaned_records,
            'retention_rate': self.retention_rate,
            'removed_duplicates': self.removed_duplicates,
            'removed_outliers': self.removed_outliers,
            'fixed_nulls': self.fixed_nulls,
            'quality_score': self.quality_score
        }

class DataCleaner:
    """
    Limpador de dados com HARDNESS máxima
    Implementa validações rigorosas e limpeza inteligente
    """
    
    def __init__(self, completeness_threshold: float = 0.95, 
                 outlier_threshold: float = 0.05,
                 config: Optional[CleaningConfig] = None):
        
        self.config = config or CleaningConfig()
        self.config.min_completeness_threshold = completeness_threshold
        self.config.max_outlier_percentage = outlier_threshold
        
        self.logger = logger.bind(component="DataCleaner")
        
        # Compilação de padrões regex
        self._compile_regex_patterns()
        
        # Cache de validações
        self._validation_cache = {}
        
        self.logger.info("DataCleaner inicializado", config=self.config.__dict__)
    
    def _compile_regex_patterns(self):
        """Compila padrões regex para performance"""
        self.regex_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^\+?[\d\s\-\(\)]{10,}$'),
            'special_chars': re.compile(r'[^\w\s@.-]'),
            'multiple_spaces': re.compile(r'\s+'),
            'currency_symbols': re.compile(r'[$€£¥₹]'),
            'numeric_only': re.compile(r'^\d+\.?\d*$')
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline principal de limpeza de dados
        
        Args:
            df: DataFrame para limpar
            
        Returns:
            DataFrame limpo e validado
        """
        if df.empty:
            self.logger.warning("DataFrame vazio fornecido para limpeza")
            return df
        
        self.logger.info("Iniciando limpeza de dados", original_records=len(df))
        
        # Inicializa relatório
        report = CleaningReport(original_records=len(df))
        
        try:
            # Fase 1: Validações iniciais
            df = self._initial_validation(df, report)
            
            # Fase 2: Limpeza de duplicatas
            df = self._remove_duplicates(df, report)
            
            # Fase 3: Tratamento de valores nulos
            df = self._handle_missing_values(df, report)
            
            # Fase 4: Padronização de strings
            df = self._standardize_strings(df, report)
            
            # Fase 5: Validação de tipos de dados
            df = self._validate_data_types(df, report)
            
            # Fase 6: Detecção e tratamento de outliers
            df = self._handle_outliers(df, report)
            
            # Fase 7: Validações específicas de domínio
            df = self._domain_specific_validations(df, report)
            
            # Fase 8: Validação final de qualidade
            self._final_quality_check(df, report)
            
            # Finaliza relatório
            report.cleaned_records = len(df)
            report.quality_score = self._calculate_quality_score(df, report)
            
            self.logger.info(
                "Limpeza de dados concluída",
                **report.cleaning_summary
            )
            
            return df
            
        except Exception as e:
            self.logger.error("Erro na limpeza de dados", error=str(e))
            raise
    
    def _initial_validation(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Validações iniciais básicas"""
        
        # Verifica colunas essenciais
        required_columns = ['user_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Colunas obrigatórias ausentes: {missing_columns}"
            report.validation_errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Remove registros com user_id nulo
        before_count = len(df)
        df = df.dropna(subset=['user_id'])
        removed = before_count - len(df)
        
        if removed > 0:
            self.logger.info("Registros com user_id nulo removidos", removed=removed)
            report.validation_errors.append(f"Removidos {removed} registros sem user_id")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Remove duplicatas com lógica inteligente"""
        
        before_count = len(df)
        
        # Estratégia 1: Duplicatas exatas
        df = df.drop_duplicates()
        exact_duplicates_removed = before_count - len(df)
        
        # Estratégia 2: Duplicatas baseadas em chaves de negócio
        if 'user_id' in df.columns and 'created_at' in df.columns:
            # Mantém o registro mais recente para duplicatas de user_id+timestamp
            df = df.sort_values('created_at', ascending=False)
            before_business_dedup = len(df)
            df = df.drop_duplicates(subset=['user_id', 'created_at'], keep='first')
            business_duplicates_removed = before_business_dedup - len(df)
        else:
            business_duplicates_removed = 0
        
        total_duplicates_removed = exact_duplicates_removed + business_duplicates_removed
        report.removed_duplicates = total_duplicates_removed
        
        # Valida threshold de duplicatas
        duplicate_percentage = (total_duplicates_removed / before_count) * 100
        if duplicate_percentage > (self.config.max_duplicate_percentage * 100):
            warning_msg = f"Alto percentual de duplicatas: {duplicate_percentage:.2f}%"
            self.logger.warning(warning_msg)
            report.validation_errors.append(warning_msg)
        
        if total_duplicates_removed > 0:
            self.logger.info(
                "Duplicatas removidas",
                exact=exact_duplicates_removed,
                business=business_duplicates_removed,
                total=total_duplicates_removed
            )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Tratamento inteligente de valores ausentes"""
        
        original_nulls = df.isnull().sum().sum()
        
        # Estratégias por tipo de coluna
        for column in df.columns:
            if column in df.columns and df[column].isnull().any():
                
                # Strings: preenchimento com 'unknown' ou moda
                if df[column].dtype == 'object':
                    if column in ['game_type', 'channel', 'status']:
                        df[column] = df[column].fillna('unknown')
                    else:
                        mode_value = df[column].mode()
                        if not mode_value.empty:
                            df[column] = df[column].fillna(mode_value[0])
                
                # Numéricos: preenchimento com mediana ou zero
                elif df[column].dtype in ['int64', 'float64']:
                    if column in ['amount', 'transaction_value']:
                        df[column] = df[column].fillna(0.0)
                    else:
                        median_value = df[column].median()
                        df[column] = df[column].fillna(median_value)
                
                # Datas: preenchimento com data padrão ou interpolação
                elif df[column].dtype == 'datetime64[ns]':
                    df[column] = df[column].fillna(datetime.now())
        
        final_nulls = df.isnull().sum().sum()
        fixed_nulls = original_nulls - final_nulls
        report.fixed_nulls = fixed_nulls
        
        # Valida completude após tratamento
        completeness = 1 - (final_nulls / (len(df) * len(df.columns)))
        if completeness < self.config.min_completeness_threshold:
            error_msg = f"Completude insuficiente: {completeness:.3f} < {self.config.min_completeness_threshold}"
            report.validation_errors.append(error_msg)
            self.logger.error(error_msg)
        
        if fixed_nulls > 0:
            self.logger.info("Valores nulos tratados", fixed=fixed_nulls, final_completeness=completeness)
        
        return df
    
    def _standardize_strings(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Padronização rigorosa de strings"""
        
        if not self.config.normalize_strings:
            return df
        
        standardized_count = 0
        
        for column in df.columns:
            if df[column].dtype == 'object':
                original_values = df[column].copy()
                
                # Conversão para string e limpeza básica
                df[column] = df[column].astype(str)
                df[column] = df[column].str.strip()
                
                # Remove caracteres especiais se configurado
                if self.config.remove_special_chars:
                    df[column] = df[column].apply(
                        lambda x: self.regex_patterns['special_chars'].sub('', x)
                    )
                
                # Normaliza espaços múltiplos
                df[column] = df[column].apply(
                    lambda x: self.regex_patterns['multiple_spaces'].sub(' ', x)
                )
                
                # Padronizações específicas por campo
                if column in ['email', 'user_email']:
                    df[column] = df[column].str.lower()
                
                elif column in ['phone', 'user_phone']:
                    df[column] = df[column].str.replace(r'[^\d+]', '', regex=True)
                
                elif column in ['game_type', 'channel']:
                    df[column] = df[column].str.lower().str.replace(' ', '_')
                
                elif column in ['currency']:
                    df[column] = self._standardize_currency(df[column])
                
                # Conta campos modificados
                changes = (original_values != df[column]).sum()
                standardized_count += changes
        
        report.standardized_fields = standardized_count
        
        if standardized_count > 0:
            self.logger.info("Campos padronizados", count=standardized_count)
        
        return df
    
    def _standardize_currency(self, series: pd.Series) -> pd.Series:
        """Padronização específica de moedas"""
        if not self.config.standardize_currencies:
            return series
        
        # Mapeamento de moedas
        currency_mapping = {
            'usd': 'USD', 'dollar': 'USD', '$': 'USD',
            'eur': 'EUR', 'euro': 'EUR', '€': 'EUR',
            'brl': 'BRL', 'real': 'BRL', 'r$': 'BRL',
            'gbp': 'GBP', 'pound': 'GBP', '£': 'GBP'
        }
        
        # Aplica mapeamento
        series = series.str.lower()
        for old, new in currency_mapping.items():
            series = series.str.replace(old, new, regex=False)
        
        return series.str.upper()
    
    def _validate_data_types(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Validação e conversão de tipos de dados"""
        
        # Mapeamento de tipos esperados
        expected_types = {
            'user_id': 'str',
            'amount': 'float64',
            'transaction_value': 'float64',
            'created_at': 'datetime64[ns]',
            'age': 'int64'
        }
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                try:
                    if expected_type == 'str':
                        df[column] = df[column].astype(str)
                    elif expected_type == 'float64':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif expected_type == 'int64':
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif expected_type == 'datetime64[ns]':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                        
                except Exception as e:
                    warning_msg = f"Erro na conversão de tipo {column}: {str(e)}"
                    self.logger.warning(warning_msg)
                    report.validation_errors.append(warning_msg)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Detecção e tratamento rigoroso de outliers"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for column in numeric_columns:
            if column in ['amount', 'transaction_value', 'age']:
                
                original_count = len(df)
                outlier_mask = pd.Series([False] * len(df))
                
                # Método 1: IQR
                if 'iqr' in self.config.outlier_methods:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.iqr_multiplier * IQR
                    upper_bound = Q3 + self.config.iqr_multiplier * IQR
                    
                    iqr_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                    outlier_mask |= iqr_outliers
                
                # Método 2: Z-Score
                if 'zscore' in self.config.outlier_methods:
                    z_scores = np.abs(stats.zscore(df[column].dropna()))
                    zscore_outliers = z_scores > self.config.zscore_threshold
                    # Expande para match com df original (tratando NaNs)
                    zscore_mask = pd.Series([False] * len(df))
                    zscore_mask.loc[df[column].dropna().index] = zscore_outliers
                    outlier_mask |= zscore_mask
                
                # Remove outliers
                df = df[~outlier_mask]
                column_outliers_removed = original_count - len(df)
                outliers_removed += column_outliers_removed
                
                if column_outliers_removed > 0:
                    self.logger.info(
                        f"Outliers removidos em {column}",
                        removed=column_outliers_removed,
                        remaining=len(df)
                    )
        
        report.removed_outliers = outliers_removed
        
        # Valida threshold de outliers
        if report.original_records > 0:
            outlier_percentage = (outliers_removed / report.original_records)
            if outlier_percentage > self.config.max_outlier_percentage:
                warning_msg = f"Alto percentual de outliers removidos: {outlier_percentage:.3f}"
                self.logger.warning(warning_msg)
                report.validation_errors.append(warning_msg)
        
        return df
    
    def _domain_specific_validations(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Validações específicas do domínio de apostas/gaming"""
        
        validation_errors = []
        
        # Validação de valores de transação
        if 'amount' in df.columns:
            # Remove valores negativos ou zero
            before_count = len(df)
            df = df[
                (df['amount'] >= self.config.min_transaction_value) & 
                (df['amount'] <= self.config.max_transaction_value)
            ]
            invalid_amounts = before_count - len(df)
            
            if invalid_amounts > 0:
                validation_errors.append(f"Removidos {invalid_amounts} registros com valores inválidos")
        
        # Validação de emails
        if self.config.email_validation and 'email' in df.columns:
            valid_emails = df['email'].apply(
                lambda x: bool(self.regex_patterns['email'].match(str(x)))
            )
            invalid_email_count = (~valid_emails).sum()
            
            if invalid_email_count > 0:
                validation_errors.append(f"{invalid_email_count} emails inválidos encontrados")
                # Remove ou marca emails inválidos
                df.loc[~valid_emails, 'email'] = 'invalid@unknown.com'
        
        # Validação de telefones
        if self.config.phone_validation and 'phone' in df.columns:
            valid_phones = df['phone'].apply(
                lambda x: bool(self.regex_patterns['phone'].match(str(x)))
            )
            invalid_phone_count = (~valid_phones).sum()
            
            if invalid_phone_count > 0:
                validation_errors.append(f"{invalid_phone_count} telefones inválidos encontrados")
                df.loc[~valid_phones, 'phone'] = 'unknown'
        
        # Validação de datas
        if 'created_at' in df.columns:
            date_mask = (
                (df['created_at'] >= self.config.min_date) & 
                (df['created_at'] <= self.config.max_date)
            )
            invalid_dates = (~date_mask).sum()
            
            if invalid_dates > 0:
                validation_errors.append(f"Removidos {invalid_dates} registros com datas inválidas")
                df = df[date_mask]
        
        # Validação de atividade mínima do usuário
        if 'user_id' in df.columns and 'created_at' in df.columns:
            user_activity = df.groupby('user_id')['created_at'].agg(['count', 'min', 'max'])
            
            # Usuários com atividade mínima
            active_users = user_activity[
                user_activity['count'] >= self.config.min_user_activity_days
            ].index
            
            before_activity_filter = len(df)
            df = df[df['user_id'].isin(active_users)]
            removed_inactive = before_activity_filter - len(df)
            
            if removed_inactive > 0:
                validation_errors.append(f"Removidos {removed_inactive} registros de usuários inativos")
        
        # Adiciona erros ao relatório
        report.validation_errors.extend(validation_errors)
        
        if validation_errors:
            self.logger.info("Validações de domínio aplicadas", errors=len(validation_errors))
        
        return df
    
    def _final_quality_check(self, df: pd.DataFrame, report: CleaningReport):
        """Verificação final de qualidade"""
        
        if df.empty:
            error_msg = "DataFrame final está vazio após limpeza"
            report.validation_errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Verifica completude final
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        if completeness < self.config.min_completeness_threshold:
            error_msg = f"Completude final insuficiente: {completeness:.3f}"
            report.validation_errors.append(error_msg)
        
        # Verifica se colunas críticas existem
        critical_columns = ['user_id']
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            error_msg = f"Colunas críticas ausentes após limpeza: {missing_critical}"
            report.validation_errors.append(error_msg)
        
        self.logger.info("Verificação final de qualidade concluída", completeness=completeness)
    
    def _calculate_quality_score(self, df: pd.DataFrame, report: CleaningReport) -> float:
        """Calcula score de qualidade dos dados"""
        
        if df.empty:
            return 0.0
        
        scores = []
        
        # Score de completude
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        scores.append(completeness)
        
        # Score de retenção
        scores.append(report.retention_rate)
        
        # Score de validação (baseado em erros)
        validation_score = max(0.0, 1.0 - (len(report.validation_errors) / 10))
        scores.append(validation_score)
        
        # Score de consistência (baseado em duplicatas)
        duplicate_score = 1.0 - min(1.0, report.removed_duplicates / max(1, report.original_records))
        scores.append(duplicate_score)
        
        # Score médio ponderado
        weights = [0.3, 0.3, 0.2, 0.2]  # Completude e retenção são mais importantes
        quality_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return round(quality_score, 3)
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validação independente de dados limpos"""
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'metrics': {},
            'recommendations': []
        }
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame está vazio")
            return validation_results
        
        # Métricas básicas
        validation_results['metrics'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Verifica thresholds
        if validation_results['metrics']['null_percentage'] > 5:
            validation_results['issues'].append("Alto percentual de valores nulos")
            validation_results['recommendations'].append("Revisar estratégias de preenchimento")
        
        if validation_results['metrics']['duplicate_percentage'] > 2:
            validation_results['issues'].append("Alto percentual de duplicatas")
            validation_results['recommendations'].append("Revisar lógica de deduplicação")
        
        # Valida se há problemas críticos
        if validation_results['issues']:
            validation_results['is_valid'] = len(validation_results['issues']) <= 2
        
        return validation_results

# Função utilitária para teste
def test_data_cleaner():
    """Teste do data cleaner"""
    
    # Cria dados de teste
    test_data = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user1', None, 'user3'],
        'amount': [100.0, -50.0, 100.0, 200.0, 50000.0],  # Inclui negativo, duplicata e outlier
        'email': ['test@email.com', 'invalid-email', 'test@email.com', 'user@test.com', None],
        'created_at': [
            datetime.now(),
            datetime.now() - timedelta(days=1),
            datetime.now(),
            None,
            datetime.now() - timedelta(days=2)
        ]
    })
    
    print("Dados originais:")
    print(test_data)
    print(f"Shape: {test_data.shape}")
    
    # Inicializa cleaner
    cleaner = DataCleaner()
    
    # Executa limpeza
    cleaned_data = cleaner.clean_data(test_data)
    
    print("\nDados limpos:")
    print(cleaned_data)
    print(f"Shape: {cleaned_data.shape}")
    
    # Valida resultado
    validation = cleaner.validate_cleaned_data(cleaned_data)
    print(f"\nValidação: {validation}")

if __name__ == "__main__":
    test_data_cleaner()