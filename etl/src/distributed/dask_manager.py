"""
🚀 Dask Distributed Manager - INDUSTRIAL HARDNESS
Manages Dask clusters for TB+/hour processing with maximum reliability

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

import os
import time
import psutil
import socket
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dask imports
import dask
from dask.distributed import Client, LocalCluster, as_completed as dask_completed
from dask.distributed import wait, fire_and_forget
from dask import delayed, compute
import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar

# Performance imports
import pandas as pd
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numba import jit, prange
import gc

# Monitoring imports  
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import memory_profiler

logger = structlog.get_logger(__name__)

# Prometheus metrics
PROCESSED_RECORDS = Counter('etl_processed_records_total', 'Total processed records', ['stage'])
PROCESSING_TIME = Histogram('etl_processing_seconds', 'Processing time', ['stage'])
ACTIVE_WORKERS = Gauge('etl_active_workers', 'Active workers count')
MEMORY_USAGE = Gauge('etl_memory_usage_bytes', 'Memory usage in bytes', ['component'])
THROUGHPUT = Gauge('etl_throughput_records_per_second', 'Records processed per second')

@dataclass
class DaskConfig:
    """Configuração otimizada para Dask industrial"""
    
    # Cluster Configuration
    n_workers: int = os.cpu_count() * 2  # Oversubscribe for I/O bound tasks
    threads_per_worker: int = 2
    memory_limit: str = 'auto'  # Will be calculated based on available RAM
    
    # Performance Optimization
    worker_class: str = 'distributed.Worker'
    scheduler_port: int = 8786
    dashboard_port: int = 8787
    
    # Advanced Settings
    spill_compression: str = 'lz4'  # Fast compression for spilling
    compression: str = 'lz4'       # Network compression
    
    # Memory Management
    memory_target_fraction: float = 0.60    # Target 60% memory usage
    memory_spill_fraction: float = 0.70     # Spill at 70%
    memory_pause_fraction: float = 0.80     # Pause at 80%
    memory_terminate_fraction: float = 0.95  # Terminate at 95%
    
    # I/O Optimization
    tcp_timeout: str = '30s'
    comm_timeout: str = '30s'
    heartbeat_interval: str = '5s'
    
    # Resilience
    allowed_failures: int = 3
    retry_attempts: int = 3
    
    # Resource Limits
    max_memory_gb: int = 64         # Maximum memory per worker
    min_memory_gb: int = 4          # Minimum memory per worker
    
    # Adaptive Scaling
    adaptive_minimum: int = 2       # Minimum workers
    adaptive_maximum: int = 100     # Maximum workers  
    adaptive_interval: str = '1s'   # Scaling check interval
    
    # Performance Monitoring
    enable_prometheus: bool = True
    prometheus_port: int = 9090

@dataclass  
class ClusterMetrics:
    """Métricas do cluster Dask"""
    workers_count: int = 0
    total_cores: int = 0
    total_memory_gb: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    throughput_records_per_sec: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    
    @property
    def efficiency_score(self) -> float:
        """Calcula score de eficiência do cluster"""
        cpu_score = min(1.0, self.cpu_utilization / 80.0)  # Target 80% CPU
        memory_score = min(1.0, self.memory_utilization / 70.0)  # Target 70% memory
        task_success_rate = self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)
        
        return (cpu_score + memory_score + task_success_rate) / 3.0

class DaskManager:
    """
    Gerenciador industrial de clusters Dask
    Otimizado para processamento de TB+/hora com máxima eficiência
    """
    
    def __init__(self, config: Optional[DaskConfig] = None):
        self.config = config or DaskConfig()
        self.logger = logger.bind(component="DaskManager")
        
        # Otimiza configuração baseada no hardware
        self._optimize_config_for_hardware()
        
        # Cluster management
        self.cluster: Optional[LocalCluster] = None
        self.client: Optional[Client] = None
        self.metrics = ClusterMetrics()
        
        # Performance tracking
        self._start_time = time.time()
        self._processed_records = 0
        self._last_throughput_check = time.time()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("DaskManager inicializado", config=self.config.__dict__)
    
    def _optimize_config_for_hardware(self):
        """Otimiza configuração baseada no hardware disponível"""
        
        # Detecta recursos do sistema
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count(logical=True)
        
        # Otimiza workers baseado na memória
        if self.config.memory_limit == 'auto':
            memory_per_worker = min(self.config.max_memory_gb, 
                                  max(self.config.min_memory_gb, 
                                      total_memory_gb / self.config.n_workers * 0.8))
            self.config.memory_limit = f"{memory_per_worker:.1f}GB"
        
        # Ajusta número de workers se necessário
        max_workers_by_memory = int(total_memory_gb / self.config.min_memory_gb)
        if self.config.n_workers > max_workers_by_memory:
            self.config.n_workers = max_workers_by_memory
            self.logger.warning(
                "Ajustando número de workers baseado na memória",
                original=self.config.n_workers,
                adjusted=max_workers_by_memory
            )
        
        # Configura Dask globalmente
        dask.config.set({
            'array.chunk-size': '256MiB',           # Chunks otimizados
            'array.slicing.split_large_chunks': True,
            'dataframe.shuffle.method': 'tasks',     # Shuffle eficiente
            'dataframe.optimize-graph': True,        # Otimização de grafos
            'distributed.worker.memory.target': self.config.memory_target_fraction,
            'distributed.worker.memory.spill': self.config.memory_spill_fraction,
            'distributed.worker.memory.pause': self.config.memory_pause_fraction,
            'distributed.worker.memory.terminate': self.config.memory_terminate_fraction,
            'distributed.comm.compression': self.config.compression,
            'distributed.worker.use-file-locking': False,  # Melhor performance
        })
        
        self.logger.info(
            "Configuração otimizada para hardware",
            total_memory_gb=total_memory_gb,
            cpu_count=cpu_count,
            workers=self.config.n_workers,
            memory_per_worker=self.config.memory_limit
        )
    
    async def start_cluster(self) -> bool:
        """Inicia cluster Dask otimizado"""
        
        try:
            self.logger.info("Iniciando cluster Dask industrial")
            
            # Configura cluster local otimizado
            self.cluster = LocalCluster(
                n_workers=self.config.n_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=self.config.memory_limit,
                scheduler_port=self.config.scheduler_port,
                dashboard_address=f':{self.config.dashboard_port}',
                worker_class=self.config.worker_class,
                
                # Otimizações de performance
                asynchronous=True,
                silence_logs=False,
                
                # Configurações de rede
                protocol='tcp',
                interface=None,
                
                # Configurações avançadas
                processes=True,  # Usa processos para better isolation
                security=None,
                host='localhost',
            )
            
            # Conecta cliente
            self.client = Client(self.cluster, asynchronous=True)
            
            # Aguarda cluster estar pronto
            await self.client.wait_for_workers(n_workers=self.config.n_workers, timeout=60)
            
            # Inicia monitoramento
            if self.config.enable_prometheus:
                self._start_prometheus_monitoring()
            
            # Inicia task de monitoramento
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Atualiza métricas iniciais
            await self._update_cluster_metrics()
            
            self.logger.info(
                "Cluster Dask iniciado com sucesso",
                workers=self.metrics.workers_count,
                cores=self.metrics.total_cores,
                memory_gb=self.metrics.total_memory_gb,
                dashboard=f"http://localhost:{self.config.dashboard_port}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Erro ao iniciar cluster Dask", error=str(e))
            await self.shutdown_cluster()
            return False
    
    def _start_prometheus_monitoring(self):
        """Inicia servidor Prometheus para métricas"""
        try:
            start_http_server(self.config.prometheus_port)
            self.logger.info(
                "Servidor Prometheus iniciado",
                port=self.config.prometheus_port
            )
        except Exception as e:
            self.logger.warning("Erro ao iniciar Prometheus", error=str(e))
    
    async def _monitoring_loop(self):
        """Loop de monitoramento contínuo"""
        
        while self.client and not self.client.status == 'closed':
            try:
                # Atualiza métricas do cluster
                await self._update_cluster_metrics()
                
                # Atualiza métricas Prometheus
                self._update_prometheus_metrics()
                
                # Verifica saúde do cluster
                await self._health_check()
                
                # Aguarda próxima verificação
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Erro no loop de monitoramento", error=str(e))
                await asyncio.sleep(10)
    
    async def _update_cluster_metrics(self):
        """Atualiza métricas do cluster"""
        
        if not self.client:
            return
        
        try:
            # Informações dos workers
            worker_info = await self.client.scheduler_info()
            
            self.metrics.workers_count = len(worker_info['workers'])
            self.metrics.total_cores = sum(w['nthreads'] for w in worker_info['workers'].values())
            self.metrics.total_memory_gb = sum(
                w['memory_limit'] for w in worker_info['workers'].values()
            ) / (1024**3)
            
            # Tasks info
            tasks_info = await self.client.scheduler_info()
            self.metrics.active_tasks = len(tasks_info.get('tasks', {}))
            
            # Calcula throughput
            current_time = time.time()
            time_delta = current_time - self._last_throughput_check
            
            if time_delta >= 1.0:  # Update every second
                self.metrics.throughput_records_per_sec = self._processed_records / time_delta
                self._processed_records = 0
                self._last_throughput_check = current_time
            
            # Utilização de recursos (aproximada)
            self.metrics.cpu_utilization = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            self.metrics.memory_utilization = memory_info.percent
            
        except Exception as e:
            self.logger.error("Erro ao atualizar métricas", error=str(e))
    
    def _update_prometheus_metrics(self):
        """Atualiza métricas Prometheus"""
        
        ACTIVE_WORKERS.set(self.metrics.workers_count)
        MEMORY_USAGE.labels(component='cluster').set(
            self.metrics.total_memory_gb * 1024**3
        )
        THROUGHPUT.set(self.metrics.throughput_records_per_sec)
    
    async def _health_check(self):
        """Verifica saúde do cluster"""
        
        if not self.client:
            return
        
        try:
            # Testa conectividade básica
            await self.client.scheduler_info()
            
            # Verifica workers inativos
            if self.metrics.workers_count < self.config.adaptive_minimum:
                self.logger.warning(
                    "Poucos workers ativos",
                    active=self.metrics.workers_count,
                    minimum=self.config.adaptive_minimum
                )
            
            # Verifica utilização de memória
            if self.metrics.memory_utilization > 90:
                self.logger.warning(
                    "Alta utilização de memória",
                    utilization=self.metrics.memory_utilization
                )
            
        except Exception as e:
            self.logger.error("Falha no health check", error=str(e))
    
    async def process_dataframe_distributed(self, 
                                          df: Union[pd.DataFrame, dd.DataFrame],
                                          process_func: callable,
                                          chunk_size: int = 1_000_000,
                                          **kwargs) -> pd.DataFrame:
        """
        Processa DataFrame de forma distribuída
        
        Args:
            df: DataFrame para processar
            process_func: Função de processamento
            chunk_size: Tamanho dos chunks
            **kwargs: Argumentos para função de processamento
            
        Returns:
            DataFrame processado
        """
        
        if not self.client:
            raise RuntimeError("Cluster Dask não está iniciado")
        
        start_time = time.time()
        
        try:
            # Converte pandas para Dask DataFrame se necessário
            if isinstance(df, pd.DataFrame):
                ddf = dd.from_pandas(df, npartitions=self.config.n_workers * 4)
            else:
                ddf = df
            
            self.logger.info(
                "Iniciando processamento distribuído",
                partitions=ddf.npartitions,
                workers=self.metrics.workers_count
            )
            
            # Aplica função de processamento em paralelo
            processed_ddf = ddf.map_partitions(
                process_func,
                **kwargs,
                meta=ddf._meta
            )
            
            # Computa resultado
            with ProgressBar():
                result_df = processed_ddf.compute()
            
            # Atualiza estatísticas
            processing_time = time.time() - start_time
            self._processed_records += len(result_df)
            
            # Atualiza métricas Prometheus
            PROCESSED_RECORDS.labels(stage='distributed_processing').inc(len(result_df))
            PROCESSING_TIME.labels(stage='distributed_processing').observe(processing_time)
            
            self.logger.info(
                "Processamento distribuído concluído",
                records=len(result_df),
                processing_time=processing_time,
                throughput=len(result_df) / processing_time
            )
            
            return result_df
            
        except Exception as e:
            self.logger.error("Erro no processamento distribuído", error=str(e))
            raise
    
    async def process_files_parallel(self, 
                                   file_paths: List[str],
                                   process_func: callable,
                                   max_workers: Optional[int] = None,
                                   **kwargs) -> List[Any]:
        """
        Processa múltiplos arquivos em paralelo
        
        Args:
            file_paths: Lista de caminhos dos arquivos
            process_func: Função de processamento
            max_workers: Máximo de workers (usa config se None)
            **kwargs: Argumentos para função de processamento
            
        Returns:
            Lista de resultados processados
        """
        
        if not self.client:
            raise RuntimeError("Cluster Dask não está iniciado")
        
        max_workers = max_workers or self.config.n_workers
        start_time = time.time()
        
        try:
            self.logger.info(
                "Iniciando processamento paralelo de arquivos",
                files=len(file_paths),
                max_workers=max_workers
            )
            
            # Cria delayed tasks
            delayed_tasks = []
            for file_path in file_paths:
                task = delayed(process_func)(file_path, **kwargs)
                delayed_tasks.append(task)
            
            # Executa em paralelo com controle de workers
            batch_size = max_workers
            results = []
            
            for i in range(0, len(delayed_tasks), batch_size):
                batch = delayed_tasks[i:i + batch_size]
                
                with ProgressBar():
                    batch_results = compute(*batch)
                
                results.extend(batch_results)
                
                # Log progresso
                self.logger.info(
                    "Batch processado",
                    batch=i // batch_size + 1,
                    total_batches=(len(delayed_tasks) + batch_size - 1) // batch_size,
                    processed_files=len(results)
                )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                "Processamento paralelo concluído",
                files_processed=len(results),
                processing_time=processing_time,
                files_per_second=len(results) / processing_time
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Erro no processamento paralelo", error=str(e))
            raise
    
    @jit(nopython=True, parallel=True)
    def _fast_numeric_processing(self, data: np.ndarray) -> np.ndarray:
        """Processamento numérico ultrarrápido com Numba"""
        result = np.empty_like(data)
        
        for i in prange(len(data)):
            # Operações numéricas otimizadas
            result[i] = data[i] * 1.1 + np.sin(data[i])
        
        return result
    
    async def optimize_memory_usage(self):
        """Otimiza uso de memória do cluster"""
        
        try:
            # Força garbage collection
            gc.collect()
            
            # Limpa cache do Dask
            if self.client:
                await self.client.run(gc.collect)
            
            # Log uso de memória
            memory_info = psutil.virtual_memory()
            self.logger.info(
                "Otimização de memória executada",
                memory_used_gb=memory_info.used / (1024**3),
                memory_available_gb=memory_info.available / (1024**3),
                memory_percent=memory_info.percent
            )
            
        except Exception as e:
            self.logger.error("Erro na otimização de memória", error=str(e))
    
    async def scale_cluster(self, target_workers: int):
        """Escala cluster dinamicamente"""
        
        if not self.cluster:
            return
        
        try:
            current_workers = self.metrics.workers_count
            
            if target_workers > current_workers:
                # Scale up
                await self.cluster.scale(target_workers)
                self.logger.info(
                    "Cluster scaled up",
                    from_workers=current_workers,
                    to_workers=target_workers
                )
            elif target_workers < current_workers:
                # Scale down
                await self.cluster.scale(target_workers)
                self.logger.info(
                    "Cluster scaled down", 
                    from_workers=current_workers,
                    to_workers=target_workers
                )
            
            # Aguarda scaling completar
            await self.client.wait_for_workers(n_workers=target_workers, timeout=30)
            
        except Exception as e:
            self.logger.error("Erro no scaling do cluster", error=str(e))
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do cluster"""
        
        status = {
            'cluster_active': self.cluster is not None,
            'client_connected': self.client is not None and self.client.status != 'closed',
            'metrics': {
                'workers': self.metrics.workers_count,
                'cores': self.metrics.total_cores,
                'memory_gb': self.metrics.total_memory_gb,
                'active_tasks': self.metrics.active_tasks,
                'throughput_rps': self.metrics.throughput_records_per_sec,
                'cpu_utilization': self.metrics.cpu_utilization,
                'memory_utilization': self.metrics.memory_utilization,
                'efficiency_score': self.metrics.efficiency_score
            },
            'uptime_seconds': time.time() - self._start_time,
            'config': self.config.__dict__
        }
        
        if self.client:
            try:
                scheduler_info = await self.client.scheduler_info()
                status['scheduler_info'] = {
                    'address': scheduler_info['address'],
                    'workers': list(scheduler_info['workers'].keys()),
                    'tasks': len(scheduler_info.get('tasks', {}))
                }
            except Exception as e:
                status['scheduler_error'] = str(e)
        
        return status
    
    async def shutdown_cluster(self):
        """Encerra cluster de forma limpa"""
        
        try:
            self.logger.info("Encerrando cluster Dask")
            
            # Cancela task de monitoramento
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Fecha cliente
            if self.client:
                await self.client.close()
                self.client = None
            
            # Fecha cluster
            if self.cluster:
                await self.cluster.close()
                self.cluster = None
            
            self.logger.info("Cluster Dask encerrado com sucesso")
            
        except Exception as e:
            self.logger.error("Erro ao encerrar cluster", error=str(e))

# Context manager para uso automático
class DaskClusterContext:
    """Context manager para cluster Dask"""
    
    def __init__(self, config: Optional[DaskConfig] = None):
        self.manager = DaskManager(config)
    
    async def __aenter__(self):
        success = await self.manager.start_cluster()
        if not success:
            raise RuntimeError("Falha ao iniciar cluster Dask")
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.shutdown_cluster()

# Função utilitária para processamento rápido
async def process_with_dask(data: Union[pd.DataFrame, List[str]], 
                           process_func: callable,
                           config: Optional[DaskConfig] = None,
                           **kwargs) -> Any:
    """
    Função utilitária para processamento com Dask
    
    Args:
        data: Dados para processar (DataFrame ou lista de arquivos)
        process_func: Função de processamento
        config: Configuração Dask
        **kwargs: Argumentos para função de processamento
        
    Returns:
        Resultado do processamento
    """
    
    async with DaskClusterContext(config) as manager:
        if isinstance(data, pd.DataFrame):
            return await manager.process_dataframe_distributed(data, process_func, **kwargs)
        elif isinstance(data, list):
            return await manager.process_files_parallel(data, process_func, **kwargs)
        else:
            raise ValueError("Tipo de dados não suportado")