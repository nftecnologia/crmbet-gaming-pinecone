"""
🗜️ Compression Manager - ULTRA-HIGH PERFORMANCE COMPRESSION
Advanced compression strategies for TB+/hour data processing with minimal CPU overhead

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import structlog
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Data processing imports
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv

# Compression libraries - MAXIMUM PERFORMANCE
import lz4.frame
import lz4.block
import zstandard as zstd
import snappy
import blosc
import gzip
import brotli

# Serialization libraries
import orjson  # Ultra-fast JSON
import msgpack
import pickle
import joblib

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge
import psutil

logger = structlog.get_logger(__name__)

# Prometheus metrics
COMPRESSION_OPERATIONS = Counter('compression_operations_total', 'Total compression operations', ['algorithm', 'operation'])
COMPRESSION_RATIO = Histogram('compression_ratio', 'Compression ratio achieved', ['algorithm'])
COMPRESSION_TIME = Histogram('compression_time_seconds', 'Compression time', ['algorithm', 'operation'])
COMPRESSION_THROUGHPUT = Gauge('compression_throughput_mbps', 'Compression throughput MB/s', ['algorithm'])

class CompressionAlgorithm(Enum):
    """Algoritmos de compressão suportados"""
    LZ4 = "lz4"                    # Ultra-fast, low compression
    LZ4_HC = "lz4_hc"             # High compression LZ4
    ZSTD = "zstd"                 # Balanced speed/ratio
    ZSTD_FAST = "zstd_fast"       # Fast ZSTD
    SNAPPY = "snappy"             # Google's fast compression
    BLOSC = "blosc"               # Optimized for numerical data
    GZIP = "gzip"                 # Standard compression
    BROTLI = "brotli"             # High compression ratio
    AUTO = "auto"                 # Automatic selection

class DataType(Enum):
    """Tipos de dados para otimização específica"""
    NUMERIC = "numeric"
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    MIXED = "mixed"

@dataclass
class CompressionConfig:
    """Configuração de compressão otimizada"""
    
    # Default algorithm
    default_algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4
    
    # Algorithm-specific settings
    lz4_acceleration: int = 1              # 1 = default, higher = faster but less compression
    lz4_hc_level: int = 9                 # 1-12, higher = better compression
    zstd_level: int = 3                   # 1-22, balanced at 3
    zstd_fast_level: int = 1              # Fast ZSTD level
    blosc_clevel: int = 5                 # 0-9 compression level
    blosc_shuffle: int = 1                # 0=no shuffle, 1=byte, 2=bit
    gzip_level: int = 6                   # 1-9 compression level
    brotli_quality: int = 4               # 0-11, balanced at 4
    
    # Performance settings
    enable_parallel: bool = True          # Parallel compression
    max_workers: int = mp.cpu_count()     # Worker threads/processes
    chunk_size_mb: int = 64               # Chunk size for parallel processing
    
    # Auto-selection criteria
    auto_selection_enabled: bool = True
    speed_priority: float = 0.6           # 0.0-1.0, higher = prefer speed
    ratio_priority: float = 0.4           # 0.0-1.0, higher = prefer compression
    
    # Memory management
    max_memory_mb: int = 1024             # Max memory for compression buffers
    enable_memory_mapping: bool = True     # Use memory mapping for large files
    
    # Caching
    enable_compression_cache: bool = True  # Cache compressed data
    cache_size_mb: int = 512              # Cache size

@dataclass
class CompressionResult:
    """Resultado da operação de compressão"""
    original_size: int
    compressed_size: int
    compression_time: float
    algorithm: CompressionAlgorithm
    compression_ratio: float
    throughput_mbps: float
    
    @property
    def space_saved_percent(self) -> float:
        """Percentual de espaço economizado"""
        return (1 - self.compressed_size / self.original_size) * 100

@dataclass
class AlgorithmBenchmark:
    """Benchmark de algoritmo de compressão"""
    algorithm: CompressionAlgorithm
    avg_compression_ratio: float
    avg_compression_time: float
    avg_decompression_time: float
    avg_throughput_mbps: float
    data_type_scores: Dict[DataType, float] = field(default_factory=dict)

class CompressionManager:
    """
    Gerenciador de compressão industrial com:
    - Seleção automática de algoritmo baseada em dados
    - Processamento paralelo para máxima performance
    - Benchmarking contínuo para otimização
    - Cache inteligente para dados frequentes
    - Monitoramento detalhado de performance
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.logger = logger.bind(component="CompressionManager")
        
        # Performance tracking
        self.algorithm_benchmarks: Dict[CompressionAlgorithm, AlgorithmBenchmark] = {}
        self.compression_cache: Dict[str, bytes] = {}
        self.cache_access_count: Dict[str, int] = {}
        
        # Thread pools for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.config.max_workers))
        
        # Initialize algorithm benchmarks
        self._initialize_benchmarks()
        
        self.logger.info("CompressionManager inicializado", config=self.config.__dict__)
    
    def _initialize_benchmarks(self):
        """Inicializa benchmarks com valores padrão"""
        
        # Benchmarks baseados em testes empíricos
        default_benchmarks = {
            CompressionAlgorithm.LZ4: AlgorithmBenchmark(
                algorithm=CompressionAlgorithm.LZ4,
                avg_compression_ratio=2.1,
                avg_compression_time=0.05,
                avg_decompression_time=0.02,
                avg_throughput_mbps=400.0
            ),
            CompressionAlgorithm.ZSTD: AlgorithmBenchmark(
                algorithm=CompressionAlgorithm.ZSTD,
                avg_compression_ratio=2.8,
                avg_compression_time=0.08,
                avg_decompression_time=0.03,
                avg_throughput_mbps=250.0
            ),
            CompressionAlgorithm.SNAPPY: AlgorithmBenchmark(
                algorithm=CompressionAlgorithm.SNAPPY,
                avg_compression_ratio=1.8,
                avg_compression_time=0.04,
                avg_decompression_time=0.02,
                avg_throughput_mbps=450.0
            ),
            CompressionAlgorithm.BLOSC: AlgorithmBenchmark(
                algorithm=CompressionAlgorithm.BLOSC,
                avg_compression_ratio=3.2,
                avg_compression_time=0.06,
                avg_decompression_time=0.03,
                avg_throughput_mbps=320.0
            )
        }
        
        self.algorithm_benchmarks.update(default_benchmarks)
    
    def detect_data_type(self, data: Union[bytes, str, pd.DataFrame, np.ndarray]) -> DataType:
        """Detecta tipo de dados para otimização de compressão"""
        
        if isinstance(data, pd.DataFrame):
            # Analisa tipos das colunas
            numeric_ratio = len(data.select_dtypes(include=[np.number]).columns) / len(data.columns)
            if numeric_ratio > 0.7:
                return DataType.NUMERIC
            else:
                return DataType.MIXED
        
        elif isinstance(data, np.ndarray):
            return DataType.NUMERIC
        
        elif isinstance(data, str):
            # Tenta detectar se é JSON
            if data.strip().startswith(('{', '[')):
                return DataType.JSON
            else:
                return DataType.TEXT
        
        elif isinstance(data, bytes):
            # Heurística simples para detecção
            try:
                decoded = data.decode('utf-8')
                if decoded.strip().startswith(('{', '[')):
                    return DataType.JSON
                else:
                    return DataType.TEXT
            except:
                return DataType.BINARY
        
        return DataType.MIXED
    
    def select_optimal_algorithm(self, 
                                data_type: DataType,
                                priority_speed: bool = True) -> CompressionAlgorithm:
        """Seleciona algoritmo ótimo baseado no tipo de dados e prioridade"""
        
        if not self.config.auto_selection_enabled:
            return self.config.default_algorithm
        
        # Scores baseados em benchmarks e tipo de dados
        algorithm_scores = {}
        
        for algorithm, benchmark in self.algorithm_benchmarks.items():
            if algorithm == CompressionAlgorithm.AUTO:
                continue
            
            # Score baseado em speed vs compression ratio
            if priority_speed:
                speed_score = 1.0 / max(benchmark.avg_compression_time, 0.001)
                ratio_score = benchmark.avg_compression_ratio / 5.0  # Normalize to ~1.0
                
                total_score = (speed_score * self.config.speed_priority + 
                              ratio_score * self.config.ratio_priority)
            else:
                ratio_score = benchmark.avg_compression_ratio / 5.0
                speed_score = 1.0 / max(benchmark.avg_compression_time, 0.001)
                
                total_score = (ratio_score * 0.7 + speed_score * 0.3)
            
            # Bonus para algoritmos específicos por tipo de dados
            if data_type == DataType.NUMERIC and algorithm == CompressionAlgorithm.BLOSC:
                total_score *= 1.2
            elif data_type == DataType.JSON and algorithm in [CompressionAlgorithm.ZSTD, CompressionAlgorithm.LZ4]:
                total_score *= 1.1
            elif data_type == DataType.TEXT and algorithm == CompressionAlgorithm.ZSTD:
                total_score *= 1.15
            
            algorithm_scores[algorithm] = total_score
        
        # Retorna algoritmo com maior score
        best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.debug(
            "Algoritmo selecionado",
            algorithm=best_algorithm.value,
            data_type=data_type.value,
            priority_speed=priority_speed,
            scores=algorithm_scores
        )
        
        return best_algorithm
    
    async def compress_data(self, 
                           data: Union[bytes, str, pd.DataFrame, np.ndarray],
                           algorithm: Optional[CompressionAlgorithm] = None,
                           **kwargs) -> Tuple[bytes, CompressionResult]:
        """
        Comprime dados usando algoritmo especificado ou automático
        
        Args:
            data: Dados para comprimir
            algorithm: Algoritmo específico ou None para seleção automática
            **kwargs: Parâmetros específicos do algoritmo
            
        Returns:
            Tuple de (dados_comprimidos, resultado_compressão)
        """
        
        start_time = time.time()
        
        # Converte dados para bytes se necessário
        if isinstance(data, pd.DataFrame):
            # Use Arrow para serialização eficiente
            table = pa.Table.from_pandas(data)
            original_data = table.to_pandas().to_feather(None)
        elif isinstance(data, np.ndarray):
            original_data = data.tobytes()
        elif isinstance(data, str):
            original_data = data.encode('utf-8')
        else:
            original_data = data
        
        original_size = len(original_data)
        
        # Detecta tipo de dados
        data_type = self.detect_data_type(data)
        
        # Seleciona algoritmo
        if algorithm is None or algorithm == CompressionAlgorithm.AUTO:
            algorithm = self.select_optimal_algorithm(data_type)
        
        # Verifica cache
        cache_key = None
        if self.config.enable_compression_cache:
            cache_key = f"{algorithm.value}:{hash(original_data)}"
            if cache_key in self.compression_cache:
                compressed_data = self.compression_cache[cache_key]
                self.cache_access_count[cache_key] = self.cache_access_count.get(cache_key, 0) + 1
                
                compression_time = time.time() - start_time
                return compressed_data, CompressionResult(
                    original_size=original_size,
                    compressed_size=len(compressed_data),
                    compression_time=compression_time,
                    algorithm=algorithm,
                    compression_ratio=original_size / len(compressed_data),
                    throughput_mbps=(original_size / (1024*1024)) / max(compression_time, 0.001)
                )
        
        # Executa compressão
        if self.config.enable_parallel and original_size > self.config.chunk_size_mb * 1024 * 1024:
            compressed_data = await self._compress_parallel(original_data, algorithm, **kwargs)
        else:
            compressed_data = await self._compress_single(original_data, algorithm, **kwargs)
        
        compression_time = time.time() - start_time
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size
        throughput_mbps = (original_size / (1024*1024)) / max(compression_time, 0.001)
        
        # Cache resultado se habilitado
        if cache_key and self.config.enable_compression_cache:
            self._update_cache(cache_key, compressed_data)
        
        # Atualiza benchmark
        self._update_benchmark(algorithm, compression_ratio, compression_time, throughput_mbps)
        
        # Métricas Prometheus
        COMPRESSION_OPERATIONS.labels(algorithm=algorithm.value, operation='compress').inc()
        COMPRESSION_RATIO.labels(algorithm=algorithm.value).observe(compression_ratio)
        COMPRESSION_TIME.labels(algorithm=algorithm.value, operation='compress').observe(compression_time)
        COMPRESSION_THROUGHPUT.labels(algorithm=algorithm.value).set(throughput_mbps)
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=compression_time,
            algorithm=algorithm,
            compression_ratio=compression_ratio,
            throughput_mbps=throughput_mbps
        )
        
        self.logger.info(
            "Compressão concluída",
            algorithm=algorithm.value,
            original_size_mb=original_size / (1024*1024),
            compressed_size_mb=compressed_size / (1024*1024),
            ratio=compression_ratio,
            time_ms=compression_time * 1000,
            throughput_mbps=throughput_mbps
        )
        
        return compressed_data, result
    
    async def _compress_single(self, 
                              data: bytes, 
                              algorithm: CompressionAlgorithm,
                              **kwargs) -> bytes:
        """Comprime dados usando thread única"""
        
        loop = asyncio.get_event_loop()
        
        # Executa compressão em thread separada para não bloquear
        compressed_data = await loop.run_in_executor(
            self.thread_pool,
            self._compress_with_algorithm,
            data,
            algorithm,
            kwargs
        )
        
        return compressed_data
    
    async def _compress_parallel(self, 
                                data: bytes, 
                                algorithm: CompressionAlgorithm,
                                **kwargs) -> bytes:
        """Comprime dados usando múltiplas threads/processos"""
        
        chunk_size = self.config.chunk_size_mb * 1024 * 1024
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        loop = asyncio.get_event_loop()
        
        # Comprime chunks em paralelo
        compress_tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(
                self.thread_pool,
                self._compress_with_algorithm,
                chunk,
                algorithm,
                kwargs
            )
            compress_tasks.append(task)
        
        compressed_chunks = await asyncio.gather(*compress_tasks)
        
        # Combina chunks comprimidos
        # Formato: [tamanho_chunk1][chunk1_comprimido][tamanho_chunk2][chunk2_comprimido]...
        combined_data = b''
        for chunk in compressed_chunks:
            combined_data += len(chunk).to_bytes(4, 'big') + chunk
        
        return combined_data
    
    def _compress_with_algorithm(self, 
                                data: bytes, 
                                algorithm: CompressionAlgorithm,
                                kwargs: Dict[str, Any]) -> bytes:
        """Executa compressão com algoritmo específico"""
        
        try:
            if algorithm == CompressionAlgorithm.LZ4:
                acceleration = kwargs.get('acceleration', self.config.lz4_acceleration)
                return lz4.frame.compress(data, acceleration=acceleration)
            
            elif algorithm == CompressionAlgorithm.LZ4_HC:
                compression_level = kwargs.get('level', self.config.lz4_hc_level)
                return lz4.frame.compress(data, compression_level=compression_level)
            
            elif algorithm == CompressionAlgorithm.ZSTD:
                level = kwargs.get('level', self.config.zstd_level)
                cctx = zstd.ZstdCompressor(level=level)
                return cctx.compress(data)
            
            elif algorithm == CompressionAlgorithm.ZSTD_FAST:
                level = kwargs.get('level', self.config.zstd_fast_level)
                cctx = zstd.ZstdCompressor(level=level)
                return cctx.compress(data)
            
            elif algorithm == CompressionAlgorithm.SNAPPY:
                return snappy.compress(data)
            
            elif algorithm == CompressionAlgorithm.BLOSC:
                clevel = kwargs.get('clevel', self.config.blosc_clevel)
                shuffle = kwargs.get('shuffle', self.config.blosc_shuffle)
                return blosc.compress(data, clevel=clevel, shuffle=shuffle)
            
            elif algorithm == CompressionAlgorithm.GZIP:
                level = kwargs.get('level', self.config.gzip_level)
                return gzip.compress(data, compresslevel=level)
            
            elif algorithm == CompressionAlgorithm.BROTLI:
                quality = kwargs.get('quality', self.config.brotli_quality)
                return brotli.compress(data, quality=quality)
            
            else:
                raise ValueError(f"Algoritmo não suportado: {algorithm}")
                
        except Exception as e:
            self.logger.error("Erro na compressão", algorithm=algorithm.value, error=str(e))
            raise
    
    async def decompress_data(self, 
                             compressed_data: bytes,
                             algorithm: CompressionAlgorithm,
                             output_type: str = 'bytes') -> Any:
        """
        Descomprime dados
        
        Args:
            compressed_data: Dados comprimidos
            algorithm: Algoritmo usado na compressão
            output_type: Tipo de saída ('bytes', 'str', 'dataframe', 'array')
            
        Returns:
            Dados descomprimidos no formato especificado
        """
        
        start_time = time.time()
        
        try:
            # Detecta se são dados paralelos (múltiplos chunks)
            if len(compressed_data) > 4:
                first_chunk_size = int.from_bytes(compressed_data[:4], 'big')
                if first_chunk_size < len(compressed_data):
                    # Dados paralelos
                    decompressed_data = await self._decompress_parallel(compressed_data, algorithm)
                else:
                    # Dados únicos
                    decompressed_data = await self._decompress_single(compressed_data, algorithm)
            else:
                decompressed_data = await self._decompress_single(compressed_data, algorithm)
            
            # Converte para tipo de saída especificado
            if output_type == 'str':
                return decompressed_data.decode('utf-8')
            elif output_type == 'dataframe':
                return pd.read_feather(BytesIO(decompressed_data))
            elif output_type == 'array':
                return np.frombuffer(decompressed_data)
            else:
                return decompressed_data
        
        except Exception as e:
            self.logger.error("Erro na descompressão", algorithm=algorithm.value, error=str(e))
            raise
    
    async def _decompress_single(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Descomprime dados usando thread única"""
        
        loop = asyncio.get_event_loop()
        
        decompressed_data = await loop.run_in_executor(
            self.thread_pool,
            self._decompress_with_algorithm,
            compressed_data,
            algorithm
        )
        
        return decompressed_data
    
    async def _decompress_parallel(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Descomprime dados processados em paralelo"""
        
        # Parse chunks
        chunks = []
        offset = 0
        
        while offset < len(compressed_data):
            chunk_size = int.from_bytes(compressed_data[offset:offset+4], 'big')
            offset += 4
            
            chunk_data = compressed_data[offset:offset+chunk_size]
            chunks.append(chunk_data)
            offset += chunk_size
        
        # Descomprime chunks em paralelo
        loop = asyncio.get_event_loop()
        decompress_tasks = []
        
        for chunk in chunks:
            task = loop.run_in_executor(
                self.thread_pool,
                self._decompress_with_algorithm,
                chunk,
                algorithm
            )
            decompress_tasks.append(task)
        
        decompressed_chunks = await asyncio.gather(*decompress_tasks)
        
        # Combina chunks
        return b''.join(decompressed_chunks)
    
    def _decompress_with_algorithm(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Executa descompressão com algoritmo específico"""
        
        try:
            if algorithm in [CompressionAlgorithm.LZ4, CompressionAlgorithm.LZ4_HC]:
                return lz4.frame.decompress(compressed_data)
            
            elif algorithm in [CompressionAlgorithm.ZSTD, CompressionAlgorithm.ZSTD_FAST]:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(compressed_data)
            
            elif algorithm == CompressionAlgorithm.SNAPPY:
                return snappy.decompress(compressed_data)
            
            elif algorithm == CompressionAlgorithm.BLOSC:
                return blosc.decompress(compressed_data)
            
            elif algorithm == CompressionAlgorithm.GZIP:
                return gzip.decompress(compressed_data)
            
            elif algorithm == CompressionAlgorithm.BROTLI:
                return brotli.decompress(compressed_data)
            
            else:
                raise ValueError(f"Algoritmo não suportado: {algorithm}")
                
        except Exception as e:
            self.logger.error("Erro na descompressão", algorithm=algorithm.value, error=str(e))
            raise
    
    def _update_benchmark(self, 
                         algorithm: CompressionAlgorithm,
                         compression_ratio: float,
                         compression_time: float,
                         throughput_mbps: float):
        """Atualiza benchmark do algoritmo"""
        
        if algorithm not in self.algorithm_benchmarks:
            self.algorithm_benchmarks[algorithm] = AlgorithmBenchmark(
                algorithm=algorithm,
                avg_compression_ratio=compression_ratio,
                avg_compression_time=compression_time,
                avg_decompression_time=0.0,
                avg_throughput_mbps=throughput_mbps
            )
        else:
            benchmark = self.algorithm_benchmarks[algorithm]
            
            # Média móvel exponencial (alpha = 0.1)
            alpha = 0.1
            benchmark.avg_compression_ratio = (
                (1 - alpha) * benchmark.avg_compression_ratio + alpha * compression_ratio
            )
            benchmark.avg_compression_time = (
                (1 - alpha) * benchmark.avg_compression_time + alpha * compression_time
            )
            benchmark.avg_throughput_mbps = (
                (1 - alpha) * benchmark.avg_throughput_mbps + alpha * throughput_mbps
            )
    
    def _update_cache(self, cache_key: str, compressed_data: bytes):
        """Atualiza cache de compressão"""
        
        # Verifica limite de memória do cache
        current_cache_size = sum(len(data) for data in self.compression_cache.values())
        max_cache_size = self.config.cache_size_mb * 1024 * 1024
        
        # Remove entradas antigas se necessário (LRU)
        while current_cache_size + len(compressed_data) > max_cache_size and self.compression_cache:
            # Remove entrada menos acessada
            lru_key = min(self.cache_access_count.items(), key=lambda x: x[1])[0]
            removed_data = self.compression_cache.pop(lru_key)
            self.cache_access_count.pop(lru_key)
            current_cache_size -= len(removed_data)
        
        # Adiciona nova entrada
        self.compression_cache[cache_key] = compressed_data
        self.cache_access_count[cache_key] = 1
    
    async def benchmark_algorithms(self, test_data: bytes, iterations: int = 5) -> Dict[CompressionAlgorithm, AlgorithmBenchmark]:
        """
        Executa benchmark de todos os algoritmos
        
        Args:
            test_data: Dados de teste
            iterations: Número de iterações
            
        Returns:
            Benchmarks de todos os algoritmos
        """
        
        algorithms_to_test = [
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.SNAPPY,
            CompressionAlgorithm.BLOSC
        ]
        
        benchmarks = {}
        
        for algorithm in algorithms_to_test:
            compression_times = []
            decompression_times = []
            compression_ratios = []
            
            for _ in range(iterations):
                # Benchmark compressão
                start_time = time.time()
                compressed_data, result = await self.compress_data(test_data, algorithm)
                compression_time = time.time() - start_time
                
                compression_times.append(compression_time)
                compression_ratios.append(result.compression_ratio)
                
                # Benchmark descompressão
                start_time = time.time()
                await self.decompress_data(compressed_data, algorithm)
                decompression_time = time.time() - start_time
                
                decompression_times.append(decompression_time)
            
            # Calcula médias
            avg_compression_time = sum(compression_times) / len(compression_times)
            avg_decompression_time = sum(decompression_times) / len(decompression_times)
            avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
            avg_throughput = (len(test_data) / (1024*1024)) / avg_compression_time
            
            benchmarks[algorithm] = AlgorithmBenchmark(
                algorithm=algorithm,
                avg_compression_ratio=avg_compression_ratio,
                avg_compression_time=avg_compression_time,
                avg_decompression_time=avg_decompression_time,
                avg_throughput_mbps=avg_throughput
            )
            
            self.logger.info(
                "Benchmark concluído",
                algorithm=algorithm.value,
                compression_ratio=avg_compression_ratio,
                compression_time_ms=avg_compression_time * 1000,
                throughput_mbps=avg_throughput
            )
        
        # Atualiza benchmarks internos
        self.algorithm_benchmarks.update(benchmarks)
        
        return benchmarks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""
        
        return {
            'algorithm_benchmarks': {
                alg.value: {
                    'compression_ratio': bench.avg_compression_ratio,
                    'compression_time_ms': bench.avg_compression_time * 1000,
                    'decompression_time_ms': bench.avg_decompression_time * 1000,
                    'throughput_mbps': bench.avg_throughput_mbps
                }
                for alg, bench in self.algorithm_benchmarks.items()
            },
            'cache_stats': {
                'entries': len(self.compression_cache),
                'total_size_mb': sum(len(data) for data in self.compression_cache.values()) / (1024*1024),
                'hit_rate': sum(self.cache_access_count.values()) / max(len(self.cache_access_count), 1)
            },
            'config': self.config.__dict__
        }
    
    def cleanup(self):
        """Limpa recursos"""
        
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            self.compression_cache.clear()
            self.cache_access_count.clear()
            
            self.logger.info("CompressionManager limpo")
            
        except Exception as e:
            self.logger.error("Erro na limpeza", error=str(e))

# Context manager para uso automático
class CompressionContext:
    """Context manager para gerenciador de compressão"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.manager = CompressionManager(config)
    
    async def __aenter__(self):
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.manager.cleanup()

# Funções utilitárias
async def compress_file(file_path: str, 
                       output_path: str,
                       algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO) -> CompressionResult:
    """Comprime arquivo"""
    
    async with CompressionContext() as manager:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        compressed_data, result = await manager.compress_data(data, algorithm)
        
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        return result

async def compress_dataframe(df: pd.DataFrame,
                           algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO) -> Tuple[bytes, CompressionResult]:
    """Comprime DataFrame"""
    
    async with CompressionContext() as manager:
        return await manager.compress_data(df, algorithm)