"""
📊 PostgreSQL Loader - Carregamento de Dados com HARDNESS Máxima
Carregamento otimizado e confiável de dados para PostgreSQL

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import structlog
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import *
import psycopg2
from psycopg2.extras import execute_values
import time
from contextlib import contextmanager
import json
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class LoaderConfig:
    """Configuração do loader PostgreSQL"""
    
    # Database Connection
    db_url: str = ""
    connection_pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Loading Strategy
    batch_size: int = 10000
    max_workers: int = 4
    enable_parallel_loading: bool = True
    
    # Table Configuration
    schema_name: str = "ml_features"
    main_table: str = "user_features"
    staging_table: str = "user_features_staging"
    backup_table: str = "user_features_backup"
    
    # Loading Options
    upsert_strategy: str = "merge"  # merge, replace, append
    create_indexes: bool = True
    create_constraints: bool = True
    enable_compression: bool = True
    
    # Quality Control
    validate_before_load: bool = True
    validate_after_load: bool = True
    enable_rollback: bool = True
    max_load_time_minutes: int = 60
    
    # Performance
    use_copy_from: bool = True
    vacuum_after_load: bool = True
    analyze_after_load: bool = True
    
    # Backup
    backup_before_load: bool = True
    retention_days: int = 30

@dataclass
class LoadingReport:
    """Relatório de carregamento"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records: int = 0
    inserted_records: int = 0
    updated_records: int = 0
    failed_records: int = 0
    loading_time_seconds: float = 0.0
    validation_passed: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.inserted_records + self.updated_records) / self.total_records
    
    @property
    def records_per_second(self) -> float:
        if self.loading_time_seconds == 0:
            return 0.0
        return self.total_records / self.loading_time_seconds

class PostgresLoader:
    """
    Loader PostgreSQL com HARDNESS máxima
    Focado em performance, confiabilidade e observabilidade
    """
    
    def __init__(self, db_url: str, batch_size: int = 10000, 
                 config: Optional[LoaderConfig] = None):
        
        self.config = config or LoaderConfig()
        self.config.db_url = db_url
        self.config.batch_size = batch_size
        
        self.logger = logger.bind(component="PostgresLoader")
        
        # Inicialização do banco
        self._initialize_database()
        
        # Cache de metadata
        self._metadata_cache = {}
        self._table_schemas = {}
        
        # Tipos de dados mapeamento
        self._type_mapping = self._setup_type_mapping()
        
        self.logger.info("PostgresLoader inicializado", config=self.config.__dict__)
    
    def _initialize_database(self):
        """Inicializa conexão com banco de dados"""
        try:
            # Cria engine com pool de conexões otimizado
            self.engine = create_engine(
                self.config.db_url,
                pool_size=self.config.connection_pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,
                echo=False,
                # Otimizações PostgreSQL
                connect_args={
                    "options": "-c default_transaction_isolation=read_committed"
                }
            )
            
            # Testa conexão
            self._test_connection()
            
            # Cria schema se não existir
            self._create_schema_if_not_exists()
            
            self.logger.info("Conexão com PostgreSQL inicializada com sucesso")
            
        except Exception as e:
            self.logger.error("Erro na inicialização do banco", error=str(e))
            raise
    
    def _test_connection(self):
        """Testa conexão com PostgreSQL"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                self.logger.info("Conexão testada com sucesso", postgres_version=version)
        except Exception as e:
            raise ConnectionError(f"Falha na conexão com PostgreSQL: {str(e)}")
    
    def _create_schema_if_not_exists(self):
        """Cria schema se não existir"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.config.schema_name}"))
                conn.commit()
                self.logger.info("Schema criado/validado", schema=self.config.schema_name)
        except Exception as e:
            self.logger.error("Erro ao criar schema", error=str(e))
            raise
    
    def _setup_type_mapping(self) -> Dict[str, type]:
        """Configura mapeamento de tipos pandas -> PostgreSQL"""
        return {
            'object': Text,
            'int64': BigInteger,
            'int32': Integer,
            'float64': Float,
            'float32': Float,
            'bool': Boolean,
            'datetime64[ns]': DateTime,
            'category': Text
        }
    
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
    
    def load_data(self, df: pd.DataFrame, table_name: Optional[str] = None,
                  execution_id: Optional[str] = None) -> int:
        """
        Carrega dados no PostgreSQL
        
        Args:
            df: DataFrame para carregar
            table_name: Nome da tabela (padrão: config.main_table)
            execution_id: ID da execução
            
        Returns:
            Número de registros carregados
        """
        if df.empty:
            self.logger.warning("DataFrame vazio fornecido para carregamento")
            return 0
        
        # Configuração
        table_name = table_name or self.config.main_table
        execution_id = execution_id or f"load_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Inicializa relatório
        report = LoadingReport(
            execution_id=execution_id,
            start_time=datetime.now(),
            total_records=len(df)
        )
        
        self.logger.info(
            "Iniciando carregamento de dados",
            execution_id=execution_id,
            table=table_name,
            records=len(df),
            strategy=self.config.upsert_strategy
        )
        
        try:
            # Fase 1: Validação pré-carregamento
            if self.config.validate_before_load:
                self._validate_data_before_load(df, report)
            
            # Fase 2: Backup da tabela existente
            if self.config.backup_before_load:
                self._create_backup(table_name, report)
            
            # Fase 3: Preparação da tabela
            self._prepare_target_table(df, table_name, report)
            
            # Fase 4: Carregamento principal
            loaded_records = self._execute_loading(df, table_name, report)
            
            # Fase 5: Pós-processamento
            self._post_loading_operations(table_name, report)
            
            # Fase 6: Validação pós-carregamento
            if self.config.validate_after_load:
                self._validate_data_after_load(table_name, report)
            
            # Finaliza relatório
            report.end_time = datetime.now()
            report.loading_time_seconds = (report.end_time - report.start_time).total_seconds()
            report.inserted_records = loaded_records
            
            self.logger.info(
                "Carregamento concluído com sucesso",
                execution_id=execution_id,
                records_loaded=loaded_records,
                loading_time=report.loading_time_seconds,
                records_per_second=report.records_per_second
            )
            
            return loaded_records
            
        except Exception as e:
            self._handle_loading_error(e, table_name, report)
            raise
    
    def _validate_data_before_load(self, df: pd.DataFrame, report: LoadingReport):
        """Valida dados antes do carregamento"""
        self.logger.info("Validando dados antes do carregamento")
        
        # Verifica se há dados
        if df.empty:
            error_msg = "DataFrame está vazio"
            report.errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Verifica colunas obrigatórias
        required_columns = ['user_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Colunas obrigatórias ausentes: {missing_columns}"
            report.errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Verifica tipos de dados
        for column in df.columns:
            if df[column].dtype == 'object':
                # Verifica se há valores muito longos
                max_length = df[column].astype(str).str.len().max()
                if max_length > 10000:  # 10KB por string
                    warning_msg = f"Coluna {column} tem valores muito longos: {max_length}"
                    report.warnings.append(warning_msg)
                    self.logger.warning(warning_msg)
        
        # Verifica duplicatas em user_id
        if df['user_id'].duplicated().any():
            duplicates = df['user_id'].duplicated().sum()
            warning_msg = f"Encontradas {duplicates} duplicatas em user_id"
            report.warnings.append(warning_msg)
            self.logger.warning(warning_msg)
        
        self.logger.info("Validação pré-carregamento concluída")
    
    def _create_backup(self, table_name: str, report: LoadingReport):
        """Cria backup da tabela existente"""
        self.logger.info("Criando backup da tabela", table=table_name)
        
        backup_table = f"{table_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with self._get_connection() as conn:
                # Verifica se tabela existe
                inspector = inspect(self.engine)
                if inspector.has_table(table_name, schema=self.config.schema_name):
                    
                    # Cria backup
                    backup_query = f"""
                    CREATE TABLE {self.config.schema_name}.{backup_table} AS 
                    SELECT * FROM {self.config.schema_name}.{table_name}
                    """
                    
                    conn.execute(text(backup_query))
                    conn.commit()
                    
                    # Limpa backups antigos
                    self._cleanup_old_backups(table_name, conn)
                    
                    self.logger.info("Backup criado com sucesso", backup_table=backup_table)
                
        except Exception as e:
            warning_msg = f"Erro ao criar backup: {str(e)}"
            report.warnings.append(warning_msg)
            self.logger.warning(warning_msg)
    
    def _cleanup_old_backups(self, table_name: str, conn):
        """Remove backups antigos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            cutoff_str = cutoff_date.strftime('%Y%m%d_%H%M%S')
            
            # Lista tabelas de backup
            inspector = inspect(self.engine)
            tables = inspector.get_table_names(schema=self.config.schema_name)
            
            backup_tables = [
                t for t in tables 
                if t.startswith(f"{table_name}_backup_") and t.split('_')[-2:] < cutoff_str.split('_')
            ]
            
            for backup_table in backup_tables:
                conn.execute(text(f"DROP TABLE IF EXISTS {self.config.schema_name}.{backup_table}"))
                self.logger.info("Backup antigo removido", table=backup_table)
                
        except Exception as e:
            self.logger.warning("Erro ao limpar backups antigos", error=str(e))
    
    def _prepare_target_table(self, df: pd.DataFrame, table_name: str, report: LoadingReport):
        """Prepara tabela de destino"""
        self.logger.info("Preparando tabela de destino", table=table_name)
        
        full_table_name = f"{self.config.schema_name}.{table_name}"
        
        try:
            with self._get_connection() as conn:
                inspector = inspect(self.engine)
                
                if not inspector.has_table(table_name, schema=self.config.schema_name):
                    # Cria tabela nova
                    self._create_table_from_dataframe(df, table_name, conn)
                else:
                    # Verifica/atualiza schema existente
                    self._validate_and_update_schema(df, table_name, conn)
                
        except Exception as e:
            error_msg = f"Erro ao preparar tabela: {str(e)}"
            report.errors.append(error_msg)
            raise
    
    def _create_table_from_dataframe(self, df: pd.DataFrame, table_name: str, conn):
        """Cria tabela baseada no DataFrame"""
        
        columns = []
        
        for column_name, dtype in df.dtypes.items():
            # Determina tipo PostgreSQL
            pg_type = self._map_pandas_type_to_postgres(dtype, df[column_name])
            
            # Configurações especiais
            nullable = True
            if column_name == 'user_id':
                nullable = False
            
            columns.append(f'"{column_name}" {pg_type}{"" if nullable else " NOT NULL"}')
        
        # Adiciona colunas de metadados
        columns.extend([
            '"created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            '"updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        ])
        
        # Cria tabela
        create_table_sql = f"""
        CREATE TABLE {self.config.schema_name}.{table_name} (
            {', '.join(columns)}
        )
        """
        
        if self.config.enable_compression:
            create_table_sql += " WITH (fillfactor=90)"
        
        conn.execute(text(create_table_sql))
        
        # Cria índices
        if self.config.create_indexes:
            self._create_indexes(table_name, conn)
        
        conn.commit()
        self.logger.info("Tabela criada com sucesso", table=table_name)
    
    def _map_pandas_type_to_postgres(self, dtype, series: pd.Series) -> str:
        """Mapeia tipo pandas para PostgreSQL"""
        
        dtype_str = str(dtype)
        
        if dtype_str.startswith('int'):
            return 'BIGINT'
        elif dtype_str.startswith('float'):
            return 'DOUBLE PRECISION'
        elif dtype_str == 'bool':
            return 'BOOLEAN'
        elif dtype_str.startswith('datetime'):
            return 'TIMESTAMP'
        elif dtype_str == 'object':
            # Determina tamanho baseado nos dados
            if series.empty:
                return 'TEXT'
            
            max_length = series.astype(str).str.len().max()
            
            if max_length <= 255:
                return f'VARCHAR({max_length + 50})'  # Buffer de segurança
            else:
                return 'TEXT'
        else:
            return 'TEXT'
    
    def _create_indexes(self, table_name: str, conn):
        """Cria índices na tabela"""
        
        indexes = [
            f'CREATE INDEX IF NOT EXISTS idx_{table_name}_user_id ON {self.config.schema_name}.{table_name} ("user_id")',
            f'CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at ON {self.config.schema_name}.{table_name} ("created_at")',
        ]
        
        # Índices específicos para features ML
        ml_columns = [
            'favorite_game_type', 'ticket_medio_categoria', 'rfm_segment',
            'customer_lifetime_value_score', 'churn_risk_score'
        ]
        
        for column in ml_columns:
            indexes.append(
                f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{column} ON {self.config.schema_name}.{table_name} ("{column}")'
            )
        
        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
            except Exception as e:
                self.logger.warning(f"Erro ao criar índice: {str(e)}")
        
        self.logger.info("Índices criados", table=table_name)
    
    def _validate_and_update_schema(self, df: pd.DataFrame, table_name: str, conn):
        """Valida e atualiza schema da tabela existente"""
        
        inspector = inspect(self.engine)
        existing_columns = {
            col['name']: col['type']
            for col in inspector.get_columns(table_name, schema=self.config.schema_name)
        }
        
        # Verifica colunas novas
        new_columns = set(df.columns) - set(existing_columns.keys())
        
        if new_columns:
            self.logger.info("Adicionando novas colunas", columns=list(new_columns))
            
            for column in new_columns:
                pg_type = self._map_pandas_type_to_postgres(df[column].dtype, df[column])
                
                alter_sql = f"""
                ALTER TABLE {self.config.schema_name}.{table_name} 
                ADD COLUMN IF NOT EXISTS "{column}" {pg_type}
                """
                
                conn.execute(text(alter_sql))
            
            conn.commit()
    
    def _execute_loading(self, df: pd.DataFrame, table_name: str, report: LoadingReport) -> int:
        """Executa o carregamento principal"""
        
        if self.config.upsert_strategy == "replace":
            return self._load_replace_strategy(df, table_name, report)
        elif self.config.upsert_strategy == "merge":
            return self._load_merge_strategy(df, table_name, report)
        elif self.config.upsert_strategy == "append":
            return self._load_append_strategy(df, table_name, report)
        else:
            raise ValueError(f"Estratégia não suportada: {self.config.upsert_strategy}")
    
    def _load_replace_strategy(self, df: pd.DataFrame, table_name: str, report: LoadingReport) -> int:
        """Estratégia de substituição completa"""
        
        self.logger.info("Executando estratégia REPLACE")
        
        with self._get_connection() as conn:
            # Trunca tabela
            conn.execute(text(f"TRUNCATE TABLE {self.config.schema_name}.{table_name}"))
            
            # Carrega dados
            loaded_records = self._bulk_insert(df, table_name, conn)
            
            conn.commit()
            
        return loaded_records
    
    def _load_merge_strategy(self, df: pd.DataFrame, table_name: str, report: LoadingReport) -> int:
        """Estratégia de merge (upsert)"""
        
        self.logger.info("Executando estratégia MERGE")
        
        staging_table = f"{table_name}_staging"
        
        with self._get_connection() as conn:
            # Cria tabela staging
            conn.execute(text(f"DROP TABLE IF EXISTS {self.config.schema_name}.{staging_table}"))
            
            conn.execute(text(f"""
            CREATE TABLE {self.config.schema_name}.{staging_table} 
            AS SELECT * FROM {self.config.schema_name}.{table_name} WHERE 1=0
            """))
            
            # Carrega dados na staging
            self._bulk_insert(df, staging_table, conn)
            
            # Executa merge
            merge_sql = f"""
            INSERT INTO {self.config.schema_name}.{table_name} 
            SELECT * FROM {self.config.schema_name}.{staging_table}
            ON CONFLICT (user_id) DO UPDATE SET
            """
            
            # Constrói lista de colunas para update
            update_columns = [
                f'"{col}" = EXCLUDED."{col}"' 
                for col in df.columns if col != 'user_id'
            ]
            update_columns.append('"updated_at" = CURRENT_TIMESTAMP')
            
            merge_sql += ', '.join(update_columns)
            
            conn.execute(text(merge_sql))
            
            # Remove staging
            conn.execute(text(f"DROP TABLE {self.config.schema_name}.{staging_table}"))
            
            conn.commit()
            
        return len(df)
    
    def _load_append_strategy(self, df: pd.DataFrame, table_name: str, report: LoadingReport) -> int:
        """Estratégia de append"""
        
        self.logger.info("Executando estratégia APPEND")
        
        with self._get_connection() as conn:
            loaded_records = self._bulk_insert(df, table_name, conn)
            conn.commit()
            
        return loaded_records
    
    def _bulk_insert(self, df: pd.DataFrame, table_name: str, conn) -> int:
        """Executa inserção em massa otimizada"""
        
        # Adiciona colunas de metadados
        df_to_load = df.copy()
        df_to_load['created_at'] = datetime.now()
        df_to_load['updated_at'] = datetime.now()
        
        if self.config.use_copy_from:
            # Usa COPY FROM para máxima performance
            return self._copy_from_insert(df_to_load, table_name, conn)
        else:
            # Usa pandas to_sql
            records_loaded = df_to_load.to_sql(
                table_name,
                conn,
                schema=self.config.schema_name,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=self.config.batch_size
            )
            
            return len(df_to_load)
    
    def _copy_from_insert(self, df: pd.DataFrame, table_name: str, conn) -> int:
        """Inserção otimizada usando COPY FROM"""
        
        # Converte DataFrame para CSV em memória
        from io import StringIO
        
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep='\\N')
        buffer.seek(0)
        
        # Usa psycopg2 raw connection para COPY
        raw_conn = conn.connection.driver_connection
        cursor = raw_conn.cursor()
        
        try:
            cursor.copy_from(
                buffer,
                f"{self.config.schema_name}.{table_name}",
                columns=list(df.columns),
                sep=',',
                null='\\N'
            )
            
            return len(df)
            
        except Exception as e:
            self.logger.error("Erro no COPY FROM", error=str(e))
            # Fallback para método normal
            return df.to_sql(
                table_name,
                conn,
                schema=self.config.schema_name,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=self.config.batch_size
            )
        finally:
            cursor.close()
    
    def _post_loading_operations(self, table_name: str, report: LoadingReport):
        """Operações pós-carregamento"""
        
        self.logger.info("Executando operações pós-carregamento")
        
        with self._get_connection() as conn:
            # VACUUM para limpeza
            if self.config.vacuum_after_load:
                conn.execute(text(f"VACUUM {self.config.schema_name}.{table_name}"))
                
            # ANALYZE para estatísticas
            if self.config.analyze_after_load:
                conn.execute(text(f"ANALYZE {self.config.schema_name}.{table_name}"))
            
            conn.commit()
    
    def _validate_data_after_load(self, table_name: str, report: LoadingReport):
        """Valida dados após carregamento"""
        
        self.logger.info("Validando dados após carregamento")
        
        with self._get_connection() as conn:
            # Conta registros
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self.config.schema_name}.{table_name}"))
            record_count = result.fetchone()[0]
            
            # Verifica se carregamento foi bem-sucedido
            if record_count == 0:
                error_msg = "Nenhum registro foi carregado"
                report.errors.append(error_msg)
                raise ValueError(error_msg)
            
            # Verifica integridade
            null_user_ids = conn.execute(text(f"""
            SELECT COUNT(*) FROM {self.config.schema_name}.{table_name} 
            WHERE user_id IS NULL
            """)).fetchone()[0]
            
            if null_user_ids > 0:
                warning_msg = f"Encontrados {null_user_ids} registros com user_id nulo"
                report.warnings.append(warning_msg)
                self.logger.warning(warning_msg)
            
            report.validation_passed = True
            self.logger.info("Validação pós-carregamento concluída", records=record_count)
    
    def _handle_loading_error(self, error: Exception, table_name: str, report: LoadingReport):
        """Trata erros de carregamento"""
        
        error_msg = f"Erro crítico no carregamento: {str(error)}"
        report.errors.append(error_msg)
        report.end_time = datetime.now()
        
        self.logger.error(
            "Carregamento falhou",
            table=table_name,
            execution_id=report.execution_id,
            error=error_msg
        )
        
        # Rollback se habilitado
        if self.config.enable_rollback:
            try:
                self._rollback_loading(table_name, report)
            except Exception as rollback_error:
                self.logger.error("Erro no rollback", error=str(rollback_error))
    
    def _rollback_loading(self, table_name: str, report: LoadingReport):
        """Executa rollback em caso de erro"""
        
        self.logger.info("Executando rollback", table=table_name)
        
        # Encontra backup mais recente
        with self._get_connection() as conn:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names(schema=self.config.schema_name)
            
            backup_tables = [
                t for t in tables 
                if t.startswith(f"{table_name}_backup_")
            ]
            
            if backup_tables:
                # Ordena e pega o mais recente
                latest_backup = sorted(backup_tables)[-1]
                
                # Restaura backup
                conn.execute(text(f"DROP TABLE IF EXISTS {self.config.schema_name}.{table_name}"))
                conn.execute(text(f"""
                CREATE TABLE {self.config.schema_name}.{table_name} AS 
                SELECT * FROM {self.config.schema_name}.{latest_backup}
                """))
                
                conn.commit()
                
                self.logger.info("Rollback executado com sucesso", backup_table=latest_backup)
    
    def get_table_stats(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtém estatísticas da tabela"""
        
        table_name = table_name or self.config.main_table
        
        try:
            with self._get_connection() as conn:
                stats_query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT user_id) as unique_users,
                    MIN(created_at) as oldest_record,
                    MAX(updated_at) as newest_record,
                    pg_size_pretty(pg_total_relation_size('{self.config.schema_name}.{table_name}')) as table_size
                FROM {self.config.schema_name}.{table_name}
                """
                
                result = conn.execute(text(stats_query)).fetchone()
                
                return {
                    'table_name': table_name,
                    'total_records': result[0],
                    'unique_users': result[1],
                    'oldest_record': result[2].isoformat() if result[2] else None,
                    'newest_record': result[3].isoformat() if result[3] else None,
                    'table_size': result[4]
                }
                
        except Exception as e:
            self.logger.error("Erro ao obter estatísticas", error=str(e))
            return {'error': str(e)}

# Função utilitária para teste
def test_postgres_loader():
    """Teste do PostgreSQL loader"""
    from dotenv import load_dotenv
    load_dotenv()
    
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/crmbet')
    
    # Cria dados de teste
    test_data = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'],
        'favorite_game_type': ['crash', 'cassino', 'esportes'],
        'ticket_medio_categoria': ['alto', 'medio', 'baixo'],
        'total_transactions': [100, 50, 25],
        'customer_lifetime_value_score': [0.8, 0.6, 0.4]
    })
    
    print("Dados de teste:")
    print(test_data)
    
    # Inicializa loader
    loader = PostgresLoader(db_url=db_url)
    
    # Executa carregamento
    try:
        loaded_records = loader.load_data(test_data, table_name='test_user_features')
        print(f"Carregados {loaded_records} registros com sucesso")
        
        # Obtém estatísticas
        stats = loader.get_table_stats('test_user_features')
        print(f"Estatísticas: {stats}")
        
    except Exception as e:
        print(f"Erro no teste: {str(e)}")

if __name__ == "__main__":
    test_postgres_loader()