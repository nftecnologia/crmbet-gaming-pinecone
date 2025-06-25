"""
🚀 Data Lake Extractor - AWS S3
Extração robusta de dados do Data Lake com HARDNESS máxima

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import os
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import logging
import io
import json
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError, NoCredentialsError
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import s3fs
from pathlib import Path
import pickle
import gzip

logger = structlog.get_logger(__name__)

@dataclass
class ExtractionConfig:
    """Configuração de extração do Data Lake"""
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region_name: str = 'us-east-1'
    
    # S3 Configuration
    bucket_name: str = 'crmbet-datalake'
    prefix: str = 'raw/'
    
    # File Processing
    supported_formats: List[str] = field(default_factory=lambda: [
        'parquet', 'csv', 'json', 'jsonl', 'pickle', 'feather'
    ])
    
    # Performance
    max_workers: int = 4
    chunk_size: int = 10000
    max_file_size_mb: int = 500
    
    # Quality Control
    min_file_age_minutes: int = 5  # Arquivo deve ter pelo menos 5 min
    max_file_age_days: int = 30    # Não processar arquivo > 30 dias
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay_seconds: int = 2
    
    # Data Filters
    date_column: str = 'created_at'
    user_activity_filters: Dict[str, Any] = field(default_factory=lambda: {
        'min_sessions': 1,
        'min_transactions': 0,
        'active_last_days': 90
    })

@dataclass
class FileMetadata:
    """Metadados de arquivo do Data Lake"""
    key: str
    size: int
    last_modified: datetime
    format: str
    estimated_records: int = 0
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.last_modified).total_seconds() / 3600
    
    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)

class DataLakeExtractor:
    """
    Extrator de dados do AWS S3 Data Lake
    Foco em performance, qualidade e observabilidade
    """
    
    def __init__(self, bucket: str, prefix: str = '', config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.config.bucket_name = bucket
        self.config.prefix = prefix
        
        self.logger = logger.bind(component="DataLakeExtractor", bucket=bucket, prefix=prefix)
        
        # Inicialização AWS
        self._initialize_aws_clients()
        
        # Cache de metadados
        self._file_cache: Dict[str, FileMetadata] = {}
        
        self.logger.info("DataLakeExtractor inicializado", config=self.config.__dict__)
    
    def _initialize_aws_clients(self):
        """Inicializa clientes AWS"""
        try:
            # Configuração de credenciais
            session_kwargs = {
                'region_name': self.config.region_name
            }
            
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key
                })
            
            # Cliente S3
            self.s3_client = boto3.client('s3', **session_kwargs)
            self.s3_resource = boto3.resource('s3', **session_kwargs)
            
            # S3FS para processamento eficiente
            self.s3fs = s3fs.S3FileSystem(
                key=self.config.aws_access_key_id,
                secret=self.config.aws_secret_access_key,
                client_kwargs={'region_name': self.config.region_name}
            )
            
            # Testa conexão
            self._test_connection()
            
            self.logger.info("Clientes AWS inicializados com sucesso")
            
        except Exception as e:
            self.logger.error("Erro na inicialização AWS", error=str(e))
            raise
    
    def _test_connection(self):
        """Testa conexão com S3"""
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
            self.logger.info("Conexão S3 testada com sucesso")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket {self.config.bucket_name} não encontrado")
            elif error_code == '403':
                raise ValueError(f"Acesso negado ao bucket {self.config.bucket_name}")
            else:
                raise
    
    def list_available_files(self, days_back: int = 7) -> List[FileMetadata]:
        """
        Lista arquivos disponíveis no Data Lake
        
        Args:
            days_back: Quantos dias atrás buscar arquivos
            
        Returns:
            Lista de metadados de arquivos
        """
        self.logger.info("Listando arquivos disponíveis", days_back=days_back)
        
        try:
            # Calcula data limite
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Lista objetos do S3
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=self.config.prefix
            )
            
            files = []
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Filtra por data
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        continue
                    
                    # Determina formato
                    file_format = self._detect_file_format(obj['Key'])
                    if file_format not in self.config.supported_formats:
                        continue
                    
                    # Filtra por tamanho
                    size_mb = obj['Size'] / (1024 * 1024)
                    if size_mb > self.config.max_file_size_mb:
                        self.logger.warning(
                            "Arquivo muito grande ignorado",
                            key=obj['Key'],
                            size_mb=size_mb
                        )
                        continue
                    
                    # Cria metadata
                    metadata = FileMetadata(
                        key=obj['Key'],
                        size=obj['Size'],
                        last_modified=obj['LastModified'].replace(tzinfo=None),
                        format=file_format
                    )
                    
                    files.append(metadata)
                    self._file_cache[obj['Key']] = metadata
            
            self.logger.info(
                "Arquivos listados com sucesso",
                total_files=len(files),
                total_size_mb=sum(f.size_mb for f in files)
            )
            
            return files
            
        except Exception as e:
            self.logger.error("Erro ao listar arquivos", error=str(e))
            raise
    
    def _detect_file_format(self, key: str) -> str:
        """Detecta formato do arquivo baseado na extensão"""
        extension = Path(key).suffix.lower().lstrip('.')
        
        format_mapping = {
            'parquet': 'parquet',
            'csv': 'csv',
            'json': 'json',
            'jsonl': 'jsonl',
            'ndjson': 'jsonl',
            'pkl': 'pickle',
            'pickle': 'pickle',
            'feather': 'feather',
            'fea': 'feather'
        }
        
        return format_mapping.get(extension, 'unknown')
    
    def extract_file(self, file_key: str, **kwargs) -> pd.DataFrame:
        """
        Extrai dados de um arquivo específico
        
        Args:
            file_key: Chave do arquivo no S3
            **kwargs: Argumentos adicionais para pandas
            
        Returns:
            DataFrame com os dados
        """
        self.logger.info("Extraindo arquivo", file_key=file_key)
        
        metadata = self._file_cache.get(file_key)
        if not metadata:
            # Busca metadados se não estiverem em cache
            try:
                response = self.s3_client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=file_key
                )
                metadata = FileMetadata(
                    key=file_key,
                    size=response['ContentLength'],
                    last_modified=response['LastModified'].replace(tzinfo=None),
                    format=self._detect_file_format(file_key)
                )
                self._file_cache[file_key] = metadata
            except ClientError as e:
                self.logger.error("Arquivo não encontrado", file_key=file_key, error=str(e))
                raise
        
        # Validações de qualidade
        self._validate_file_metadata(metadata)
        
        # Extração baseada no formato
        try:
            s3_path = f"s3://{self.config.bucket_name}/{file_key}"
            
            if metadata.format == 'parquet':
                df = self._extract_parquet(s3_path, **kwargs)
            elif metadata.format == 'csv':
                df = self._extract_csv(s3_path, **kwargs)
            elif metadata.format in ['json', 'jsonl']:
                df = self._extract_json(s3_path, metadata.format, **kwargs)
            elif metadata.format == 'pickle':
                df = self._extract_pickle(s3_path, **kwargs)
            elif metadata.format == 'feather':
                df = self._extract_feather(s3_path, **kwargs)
            else:
                raise ValueError(f"Formato não suportado: {metadata.format}")
            
            # Validações pós-extração
            df = self._post_extraction_validation(df, metadata)
            
            self.logger.info(
                "Arquivo extraído com sucesso",
                file_key=file_key,
                records=len(df),
                columns=len(df.columns),
                memory_mb=df.memory_usage(deep=True).sum() / (1024*1024)
            )
            
            return df
            
        except Exception as e:
            self.logger.error("Erro na extração do arquivo", file_key=file_key, error=str(e))
            raise
    
    def _validate_file_metadata(self, metadata: FileMetadata):
        """Valida metadados do arquivo"""
        # Verifica idade mínima (arquivo não deve estar sendo escrito)
        if metadata.age_hours < (self.config.min_file_age_minutes / 60):
            raise ValueError(f"Arquivo muito recente: {metadata.key}")
        
        # Verifica idade máxima
        if metadata.age_hours > (self.config.max_file_age_days * 24):
            raise ValueError(f"Arquivo muito antigo: {metadata.key}")
        
        # Verifica tamanho
        if metadata.size_mb > self.config.max_file_size_mb:
            raise ValueError(f"Arquivo muito grande: {metadata.key} ({metadata.size_mb:.1f}MB)")
    
    def _extract_parquet(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Extrai arquivo Parquet"""
        return pd.read_parquet(s3_path, storage_options={
            'key': self.config.aws_access_key_id,
            'secret': self.config.aws_secret_access_key
        }, **kwargs)
    
    def _extract_csv(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Extrai arquivo CSV"""
        default_kwargs = {
            'encoding': 'utf-8',
            'na_values': ['', 'NULL', 'null', 'None', 'nan'],
            'keep_default_na': True,
            'chunksize': None  # Processamento completo
        }
        default_kwargs.update(kwargs)
        
        return pd.read_csv(s3_path, storage_options={
            'key': self.config.aws_access_key_id,
            'secret': self.config.aws_secret_access_key
        }, **default_kwargs)
    
    def _extract_json(self, s3_path: str, format_type: str, **kwargs) -> pd.DataFrame:
        """Extrai arquivo JSON/JSONL"""
        with self.s3fs.open(s3_path, 'r') as f:
            if format_type == 'json':
                data = json.load(f)
                return pd.json_normalize(data)
            else:  # jsonl
                lines = []
                for line in f:
                    lines.append(json.loads(line.strip()))
                return pd.json_normalize(lines)
    
    def _extract_pickle(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Extrai arquivo Pickle"""
        with self.s3fs.open(s3_path, 'rb') as f:
            return pickle.load(f)
    
    def _extract_feather(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Extrai arquivo Feather"""
        return pd.read_feather(s3_path, storage_options={
            'key': self.config.aws_access_key_id,
            'secret': self.config.aws_secret_access_key
        }, **kwargs)
    
    def _post_extraction_validation(self, df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
        """Validações pós-extração"""
        
        # Verifica se DataFrame não está vazio
        if df.empty:
            raise ValueError(f"Arquivo extraído está vazio: {metadata.key}")
        
        # Verifica colunas essenciais
        required_columns = ['user_id']
        if self.config.date_column:
            required_columns.append(self.config.date_column)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(
                "Colunas obrigatórias ausentes",
                file_key=metadata.key,
                missing=missing_columns
            )
        
        # Converte data se presente
        if self.config.date_column in df.columns:
            df[self.config.date_column] = pd.to_datetime(df[self.config.date_column], errors='coerce')
        
        # Remove registros com user_id nulo
        if 'user_id' in df.columns:
            before_count = len(df)
            df = df.dropna(subset=['user_id'])
            after_count = len(df)
            
            if before_count != after_count:
                self.logger.info(
                    "Registros com user_id nulo removidos",
                    file_key=metadata.key,
                    removed=before_count - after_count
                )
        
        # Atualiza estimativa de registros no metadata
        metadata.estimated_records = len(df)
        
        return df
    
    def extract_all(self, days_back: int = 7, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Extrai todos os arquivos disponíveis
        
        Args:
            days_back: Quantos dias atrás buscar
            max_files: Máximo de arquivos a processar
            
        Returns:
            DataFrame combinado de todos os arquivos
        """
        self.logger.info("Iniciando extração completa", days_back=days_back, max_files=max_files)
        
        # Lista arquivos
        files = self.list_available_files(days_back)
        
        if max_files:
            files = files[:max_files]
        
        if not files:
            self.logger.warning("Nenhum arquivo encontrado para extração")
            return pd.DataFrame()
        
        # Extração paralela
        dataframes = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submete todas as extrações
            future_to_file = {
                executor.submit(self.extract_file, file.key): file 
                for file in files
            }
            
            # Coleta resultados
            for future in as_completed(future_to_file):
                file_metadata = future_to_file[future]
                
                try:
                    df = future.result()
                    if not df.empty:
                        # Adiciona metadados ao DataFrame
                        df['_source_file'] = file_metadata.key
                        df['_extraction_timestamp'] = datetime.now()
                        dataframes.append(df)
                        
                except Exception as e:
                    self.logger.error(
                        "Erro na extração de arquivo",
                        file_key=file_metadata.key,
                        error=str(e)
                    )
                    # Continua com outros arquivos em caso de erro
        
        if not dataframes:
            self.logger.warning("Nenhum arquivo foi extraído com sucesso")
            return pd.DataFrame()
        
        # Combina todos os DataFrames
        combined_df = self._combine_dataframes(dataframes)
        
        # Aplicar filtros de qualidade
        filtered_df = self._apply_quality_filters(combined_df)
        
        self.logger.info(
            "Extração completa finalizada",
            files_processed=len(dataframes),
            total_records=len(filtered_df),
            memory_mb=filtered_df.memory_usage(deep=True).sum() / (1024*1024)
        )
        
        return filtered_df
    
    def _combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combina múltiplos DataFrames de forma inteligente"""
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Verifica compatibilidade de schemas
        schemas = [set(df.columns) for df in dataframes]
        common_columns = set.intersection(*schemas)
        
        if len(common_columns) < 3:  # Mínimo de colunas comuns
            self.logger.warning(
                "Poucos campos comuns entre arquivos",
                common_columns=len(common_columns)
            )
        
        # Normaliza colunas
        normalized_dfs = []
        for df in dataframes:
            # Mantém apenas colunas comuns
            df_normalized = df[list(common_columns)]
            normalized_dfs.append(df_normalized)
        
        # Combina com concatenação
        combined_df = pd.concat(normalized_dfs, ignore_index=True, sort=False)
        
        # Remove duplicatas baseado em user_id e timestamp
        if 'user_id' in combined_df.columns and self.config.date_column in combined_df.columns:
            before_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(
                subset=['user_id', self.config.date_column],
                keep='last'
            )
            after_count = len(combined_df)
            
            if before_count != after_count:
                self.logger.info(
                    "Duplicatas removidas",
                    removed=before_count - after_count
                )
        
        return combined_df
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica filtros de qualidade aos dados"""
        
        original_count = len(df)
        
        # Filtro de atividade do usuário
        if 'user_id' in df.columns:
            # Remove usuários com poucas sessões/transações
            user_activity = df.groupby('user_id').size()
            active_users = user_activity[
                user_activity >= self.config.user_activity_filters['min_sessions']
            ].index
            
            df = df[df['user_id'].isin(active_users)]
        
        # Filtro temporal
        if self.config.date_column in df.columns:
            cutoff_date = datetime.now() - timedelta(
                days=self.config.user_activity_filters['active_last_days']
            )
            df = df[df[self.config.date_column] >= cutoff_date]
        
        final_count = len(df)
        
        if original_count != final_count:
            self.logger.info(
                "Filtros de qualidade aplicados",
                original_records=original_count,
                filtered_records=final_count,
                removed=original_count - final_count
            )
        
        return df
    
    def extract_recent(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Extrai dados recentes
        
        Args:
            hours_back: Horas atrás para buscar dados
            
        Returns:
            DataFrame com dados recentes
        """
        days_back = max(1, hours_back // 24 + 1)
        
        # Extrai dados
        df = self.extract_all(days_back=days_back)
        
        if df.empty or self.config.date_column not in df.columns:
            return df
        
        # Filtra por tempo
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_df = df[df[self.config.date_column] >= cutoff_time]
        
        self.logger.info(
            "Dados recentes extraídos",
            hours_back=hours_back,
            total_records=len(df),
            recent_records=len(recent_df)
        )
        
        return recent_df
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de extração"""
        
        cached_files = list(self._file_cache.values())
        
        if not cached_files:
            return {"status": "no_files_cached"}
        
        stats = {
            "total_files": len(cached_files),
            "total_size_mb": sum(f.size_mb for f in cached_files),
            "formats": {},
            "avg_file_age_hours": sum(f.age_hours for f in cached_files) / len(cached_files),
            "oldest_file_hours": max(f.age_hours for f in cached_files),
            "newest_file_hours": min(f.age_hours for f in cached_files)
        }
        
        # Estatísticas por formato
        for file in cached_files:
            if file.format not in stats["formats"]:
                stats["formats"][file.format] = {"count": 0, "size_mb": 0}
            stats["formats"][file.format]["count"] += 1
            stats["formats"][file.format]["size_mb"] += file.size_mb
        
        return stats

# Função utilitária para teste
def test_extraction():
    """Função de teste da extração"""
    from dotenv import load_dotenv
    load_dotenv()
    
    config = ExtractionConfig(
        bucket_name=os.getenv('DATA_LAKE_BUCKET', 'test-bucket'),
        prefix=os.getenv('DATA_LAKE_PREFIX', 'raw/'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    extractor = DataLakeExtractor(
        bucket=config.bucket_name,
        prefix=config.prefix,
        config=config
    )
    
    # Testa listagem
    files = extractor.list_available_files(days_back=1)
    print(f"Arquivos encontrados: {len(files)}")
    
    # Testa extração
    if files:
        df = extractor.extract_file(files[0].key)
        print(f"Primeiro arquivo extraído: {len(df)} registros")
    
    # Estatísticas
    stats = extractor.get_extraction_stats()
    print(f"Estatísticas: {stats}")

if __name__ == "__main__":
    test_extraction()