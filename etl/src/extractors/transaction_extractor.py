"""
💳 Transaction Extractor - PostgreSQL tbl_transactions
Extração inteligente de dados transacionais com HARDNESS máxima

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import psycopg2
from psycopg2.extras import RealDictCursor
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from contextlib import contextmanager

logger = structlog.get_logger(__name__)

@dataclass
class TransactionExtractionConfig:
    """Configuração para extração de transações"""
    # Database
    db_url: str = ""
    connection_pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    # Query Configuration
    batch_size: int = 50000
    max_days_back: int = 90
    
    # Table Configuration
    table_name: str = "tbl_transactions"
    primary_key: str = "transaction_id"
    user_id_column: str = "user_id"
    amount_column: str = "amount"
    type_column: str = "transaction_type"
    status_column: str = "status"
    timestamp_column: str = "created_at"
    
    # Data Filters
    valid_statuses: List[str] = field(default_factory=lambda: [
        'completed', 'success', 'approved', 'confirmed'
    ])
    valid_types: List[str] = field(default_factory=lambda: [
        'deposit', 'withdrawal', 'bet', 'win', 'bonus', 'refund'
    ])
    min_amount: float = 0.01
    max_amount: float = 100000.0
    
    # Performance
    enable_parallel_queries: bool = True
    query_timeout_seconds: int = 300
    
    # Quality Control
    required_columns: List[str] = field(default_factory=lambda: [
        'user_id', 'amount', 'transaction_type', 'status', 'created_at'
    ])

@dataclass
class TransactionStats:
    """Estatísticas de transações extraídas"""
    total_transactions: int = 0
    unique_users: int = 0
    total_amount: float = 0.0
    date_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.min, datetime.max))
    type_distribution: Dict[str, int] = field(default_factory=dict)
    status_distribution: Dict[str, int] = field(default_factory=dict)
    avg_transaction_value: float = 0.0
    
    def calculate_derived_stats(self, df: pd.DataFrame):
        """Calcula estatísticas derivadas"""
        if df.empty:
            return
        
        self.total_transactions = len(df)
        self.unique_users = df['user_id'].nunique()
        self.total_amount = df['amount'].sum()
        self.avg_transaction_value = df['amount'].mean()
        
        if 'created_at' in df.columns:
            self.date_range = (df['created_at'].min(), df['created_at'].max())
        
        if 'transaction_type' in df.columns:
            self.type_distribution = df['transaction_type'].value_counts().to_dict()
        
        if 'status' in df.columns:
            self.status_distribution = df['status'].value_counts().to_dict()

class TransactionExtractor:
    """
    Extrator de dados transacionais do PostgreSQL
    Focado em performance, qualidade e integridade de dados
    """
    
    def __init__(self, db_url: str, config: Optional[TransactionExtractionConfig] = None):
        self.config = config or TransactionExtractionConfig()
        self.config.db_url = db_url
        
        self.logger = logger.bind(component="TransactionExtractor")
        
        # Inicialização do banco
        self._initialize_database()
        
        # Cache de schema
        self._schema_cache = {}
        
        # Estatísticas
        self.last_extraction_stats: Optional[TransactionStats] = None
        
        self.logger.info("TransactionExtractor inicializado", config=self.config.__dict__)
    
    def _initialize_database(self):
        """Inicializa conexão com banco de dados"""
        try:
            # Cria engine com pool de conexões
            self.engine = create_engine(
                self.config.db_url,
                pool_size=self.config.connection_pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_pre_ping=True,  # Verifica conexões antes de usar
                echo=False  # Log SQL queries (desabilitado para performance)
            )
            
            # Testa conexão
            self._test_connection()
            
            # Valida schema da tabela
            self._validate_table_schema()
            
            self.logger.info("Conexão com banco inicializada com sucesso")
            
        except Exception as e:
            self.logger.error("Erro na inicialização do banco", error=str(e))
            raise
    
    def _test_connection(self):
        """Testa conexão com banco"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            self.logger.info("Conexão testada com sucesso")
        except Exception as e:
            raise ConnectionError(f"Falha na conexão com banco: {str(e)}")
    
    def _validate_table_schema(self):
        """Valida schema da tabela de transações"""
        try:
            inspector = inspect(self.engine)
            
            # Verifica se tabela existe
            if not inspector.has_table(self.config.table_name):
                raise ValueError(f"Tabela {self.config.table_name} não encontrada")
            
            # Obtém colunas da tabela
            columns = inspector.get_columns(self.config.table_name)
            column_names = [col['name'] for col in columns]
            
            # Verifica colunas obrigatórias
            missing_columns = [
                col for col in self.config.required_columns 
                if col not in column_names
            ]
            
            if missing_columns:
                self.logger.warning(
                    "Colunas obrigatórias ausentes",
                    missing=missing_columns,
                    available=column_names
                )
            
            # Cache do schema
            self._schema_cache[self.config.table_name] = {
                'columns': column_names,
                'column_info': columns
            }
            
            self.logger.info(
                "Schema da tabela validado",
                table=self.config.table_name,
                columns=len(column_names)
            )
            
        except Exception as e:
            self.logger.error("Erro na validação do schema", error=str(e))
            raise
    
    @contextmanager
    def _get_connection(self):
        """Context manager para conexões"""
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        finally:
            if connection:
                connection.close()
    
    def extract_recent(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Extrai transações recentes
        
        Args:
            hours_back: Horas atrás para buscar transações
            
        Returns:
            DataFrame com transações recentes
        """
        self.logger.info("Extraindo transações recentes", hours_back=hours_back)
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return self.extract_by_date_range(
            start_date=cutoff_time,
            end_date=datetime.now()
        )
    
    def extract_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extrai transações por intervalo de datas
        
        Args:
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            DataFrame com transações do período
        """
        self.logger.info(
            "Extraindo transações por período",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        # Constrói query otimizada
        query = self._build_date_range_query(start_date, end_date)
        
        # Executa extração
        df = self._execute_extraction_query(query, {
            'start_date': start_date,
            'end_date': end_date
        })
        
        return df
    
    def extract_by_user_ids(self, user_ids: List[str], days_back: int = 30) -> pd.DataFrame:
        """
        Extrai transações de usuários específicos
        
        Args:
            user_ids: Lista de IDs de usuários
            days_back: Dias atrás para buscar
            
        Returns:
            DataFrame com transações dos usuários
        """
        if not user_ids:
            return pd.DataFrame()
        
        self.logger.info(
            "Extraindo transações por usuários",
            user_count=len(user_ids),
            days_back=days_back
        )
        
        # Limita data
        start_date = datetime.now() - timedelta(days=days_back)
        
        # Processa em batches para evitar queries muito grandes
        batch_size = 1000
        user_batches = [
            user_ids[i:i + batch_size] 
            for i in range(0, len(user_ids), batch_size)
        ]
        
        dataframes = []
        
        for batch in user_batches:
            query = self._build_user_batch_query(batch, start_date)
            df_batch = self._execute_extraction_query(query, {
                'user_ids': tuple(batch),
                'start_date': start_date
            })
            
            if not df_batch.empty:
                dataframes.append(df_batch)
        
        if not dataframes:
            return pd.DataFrame()
        
        # Combina todos os batches
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        self.logger.info(
            "Extração por usuários concluída",
            total_transactions=len(combined_df),
            unique_users=combined_df['user_id'].nunique() if not combined_df.empty else 0
        )
        
        return combined_df
    
    def extract_all_incremental(self, last_extraction_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Extração incremental baseada em timestamp
        
        Args:
            last_extraction_time: Última vez que dados foram extraídos
            
        Returns:
            DataFrame com novos dados desde a última extração
        """
        if last_extraction_time is None:
            # Se não há timestamp anterior, busca últimas 24h
            last_extraction_time = datetime.now() - timedelta(hours=24)
        
        self.logger.info(
            "Executando extração incremental",
            last_extraction=last_extraction_time.isoformat()
        )
        
        return self.extract_by_date_range(
            start_date=last_extraction_time,
            end_date=datetime.now()
        )
    
    def _build_date_range_query(self, start_date: datetime, end_date: datetime) -> str:
        """Constrói query para extração por período"""
        
        # Query base otimizada
        query = f"""
        SELECT 
            {self.config.primary_key},
            {self.config.user_id_column},
            {self.config.amount_column},
            {self.config.type_column},
            {self.config.status_column},
            {self.config.timestamp_column},
            -- Campos adicionais para análise
            EXTRACT(HOUR FROM {self.config.timestamp_column}) as transaction_hour,
            EXTRACT(DOW FROM {self.config.timestamp_column}) as day_of_week,
            DATE({self.config.timestamp_column}) as transaction_date
        FROM {self.config.table_name}
        WHERE {self.config.timestamp_column} >= :start_date
          AND {self.config.timestamp_column} <= :end_date
          AND {self.config.status_column} = ANY(:valid_statuses)
          AND {self.config.type_column} = ANY(:valid_types)
          AND {self.config.amount_column} >= :min_amount
          AND {self.config.amount_column} <= :max_amount
          AND {self.config.user_id_column} IS NOT NULL
        ORDER BY {self.config.timestamp_column} DESC
        """
        
        return query
    
    def _build_user_batch_query(self, user_ids: List[str], start_date: datetime) -> str:
        """Constrói query para batch de usuários"""
        
        query = f"""
        SELECT 
            {self.config.primary_key},
            {self.config.user_id_column},
            {self.config.amount_column},
            {self.config.type_column},
            {self.config.status_column},
            {self.config.timestamp_column},
            EXTRACT(HOUR FROM {self.config.timestamp_column}) as transaction_hour,
            EXTRACT(DOW FROM {self.config.timestamp_column}) as day_of_week,
            DATE({self.config.timestamp_column}) as transaction_date
        FROM {self.config.table_name}
        WHERE {self.config.user_id_column} = ANY(:user_ids)
          AND {self.config.timestamp_column} >= :start_date
          AND {self.config.status_column} = ANY(:valid_statuses)
          AND {self.config.type_column} = ANY(:valid_types)
          AND {self.config.amount_column} >= :min_amount
          AND {self.config.amount_column} <= :max_amount
        ORDER BY {self.config.user_id_column}, {self.config.timestamp_column} DESC
        """
        
        return query
    
    def _execute_extraction_query(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Executa query de extração com parâmetros
        
        Args:
            query: Query SQL
            params: Parâmetros da query
            
        Returns:
            DataFrame com resultados
        """
        start_time = time.time()
        
        # Adiciona parâmetros de filtro padrão
        default_params = {
            'valid_statuses': self.config.valid_statuses,
            'valid_types': self.config.valid_types,
            'min_amount': self.config.min_amount,
            'max_amount': self.config.max_amount
        }
        default_params.update(params)
        
        try:
            with self._get_connection() as conn:
                # Usa pandas read_sql para eficiência
                df = pd.read_sql(
                    query,
                    conn,
                    params=default_params,
                    parse_dates=[self.config.timestamp_column]
                )
            
            execution_time = time.time() - start_time
            
            # Post-processamento
            df = self._post_process_dataframe(df)
            
            # Calcula estatísticas
            stats = TransactionStats()
            stats.calculate_derived_stats(df)
            self.last_extraction_stats = stats
            
            self.logger.info(
                "Query executada com sucesso",
                records=len(df),
                execution_time=execution_time,
                unique_users=stats.unique_users,
                total_amount=stats.total_amount
            )
            
            return df
            
        except Exception as e:
            self.logger.error("Erro na execução da query", error=str(e), query=query[:200])
            raise
    
    def _post_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processamento do DataFrame extraído"""
        
        if df.empty:
            return df
        
        # Renomeia colunas para padrão
        column_mapping = {
            self.config.user_id_column: 'user_id',
            self.config.amount_column: 'amount',
            self.config.type_column: 'transaction_type',
            self.config.status_column: 'status',
            self.config.timestamp_column: 'created_at'
        }
        
        # Renomeia apenas colunas que existem
        existing_mappings = {
            old: new for old, new in column_mapping.items() 
            if old in df.columns and old != new
        }
        
        if existing_mappings:
            df = df.rename(columns=existing_mappings)
        
        # Converte tipos de dados
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(str)
        
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Remove registros com dados críticos nulos
        before_count = len(df)
        df = df.dropna(subset=['user_id', 'amount'])
        after_count = len(df)
        
        if before_count != after_count:
            self.logger.info(
                "Registros com dados nulos removidos",
                removed=before_count - after_count
            )
        
        # Adiciona campos calculados
        if 'created_at' in df.columns:
            df['extraction_timestamp'] = datetime.now()
            
            # Campos de tempo úteis para ML
            df['hour_of_day'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            df['month'] = df['created_at'].dt.month
            df['quarter'] = df['created_at'].dt.quarter
        
        return df
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas da tabela de transações"""
        
        try:
            with self._get_connection() as conn:
                # Query para estatísticas básicas
                stats_query = f"""
                SELECT 
                    COUNT(*) as total_transactions,
                    COUNT(DISTINCT {self.config.user_id_column}) as unique_users,
                    SUM({self.config.amount_column}) as total_amount,
                    AVG({self.config.amount_column}) as avg_amount,
                    MIN({self.config.timestamp_column}) as oldest_transaction,
                    MAX({self.config.timestamp_column}) as newest_transaction,
                    COUNT(*) FILTER (WHERE {self.config.timestamp_column} >= NOW() - INTERVAL '24 hours') as last_24h_count
                FROM {self.config.table_name}
                WHERE {self.config.status_column} = ANY(:valid_statuses)
                """
                
                result = conn.execute(text(stats_query), {
                    'valid_statuses': self.config.valid_statuses
                }).fetchone()
                
                # Query para distribuição por tipo
                type_dist_query = f"""
                SELECT {self.config.type_column}, COUNT(*) as count
                FROM {self.config.table_name}
                WHERE {self.config.status_column} = ANY(:valid_statuses)
                GROUP BY {self.config.type_column}
                ORDER BY count DESC
                """
                
                type_results = conn.execute(text(type_dist_query), {
                    'valid_statuses': self.config.valid_statuses
                }).fetchall()
                
                stats = {
                    'total_transactions': result[0],
                    'unique_users': result[1],
                    'total_amount': float(result[2]) if result[2] else 0.0,
                    'avg_amount': float(result[3]) if result[3] else 0.0,
                    'oldest_transaction': result[4].isoformat() if result[4] else None,
                    'newest_transaction': result[5].isoformat() if result[5] else None,
                    'last_24h_count': result[6],
                    'type_distribution': {row[0]: row[1] for row in type_results}
                }
                
                self.logger.info("Estatísticas da tabela obtidas", stats=stats)
                return stats
                
        except Exception as e:
            self.logger.error("Erro ao obter estatísticas", error=str(e))
            return {'error': str(e)}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida qualidade dos dados extraídos"""
        
        if df.empty:
            return {'status': 'empty', 'issues': ['DataFrame vazio']}
        
        issues = []
        metrics = {}
        
        # Verifica completude
        null_counts = df.isnull().sum()
        total_rows = len(df)
        
        for column in ['user_id', 'amount', 'created_at']:
            if column in df.columns:
                null_pct = (null_counts[column] / total_rows) * 100
                metrics[f'{column}_null_pct'] = null_pct
                
                if null_pct > 1:  # Mais de 1% nulo
                    issues.append(f'{column}: {null_pct:.2f}% valores nulos')
        
        # Verifica valores de amount
        if 'amount' in df.columns:
            zero_amounts = (df['amount'] == 0).sum()
            negative_amounts = (df['amount'] < 0).sum()
            
            metrics['zero_amounts'] = zero_amounts
            metrics['negative_amounts'] = negative_amounts
            
            if zero_amounts > 0:
                issues.append(f'{zero_amounts} transações com valor zero')
            
            if negative_amounts > 0:
                issues.append(f'{negative_amounts} transações com valor negativo')
        
        # Verifica duplicatas
        if 'user_id' in df.columns and 'created_at' in df.columns:
            duplicates = df.duplicated(subset=['user_id', 'created_at']).sum()
            metrics['duplicates'] = duplicates
            
            if duplicates > 0:
                issues.append(f'{duplicates} possíveis transações duplicadas')
        
        status = 'good' if not issues else 'warnings' if len(issues) < 3 else 'critical'
        
        return {
            'status': status,
            'issues': issues,
            'metrics': metrics,
            'total_records': total_rows
        }

# Função utilitária para teste
def test_extraction():
    """Função de teste da extração"""
    from dotenv import load_dotenv
    load_dotenv()
    
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/crmbet')
    
    extractor = TransactionExtractor(db_url=db_url)
    
    # Testa estatísticas da tabela
    stats = extractor.get_table_stats()
    print(f"Estatísticas da tabela: {stats}")
    
    # Testa extração recente
    df = extractor.extract_recent(hours_back=24)
    print(f"Transações últimas 24h: {len(df)} registros")
    
    if not df.empty:
        # Valida qualidade
        quality = extractor.validate_data_quality(df)
        print(f"Qualidade dos dados: {quality['status']}")
        
        if quality['issues']:
            print(f"Problemas encontrados: {quality['issues']}")

if __name__ == "__main__":
    test_extraction()