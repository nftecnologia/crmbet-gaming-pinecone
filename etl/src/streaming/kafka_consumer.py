"""
🚀 Kafka Stream Consumer - INDUSTRIAL REAL-TIME PROCESSING
High-throughput Kafka consumer for TB+/hour streaming with <5min latency

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

import os
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import concurrent.futures
import structlog

# Kafka imports
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError, KafkaTimeoutError, CommitFailedError
from confluent_kafka import Consumer as ConfluentConsumer, OFFSET_BEGINNING, OFFSET_END
from confluent_kafka.admin import AdminClient, NewTopic
import aiokafka

# Data processing imports
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from collections import deque
import orjson  # Ultra-fast JSON
import msgpack  # Fast serialization
import lz4.frame  # Fast compression

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge
import psutil
import memory_profiler

# Circuit breaker imports
from pybreaker import CircuitBreaker
import tenacity
from tenacity import retry, wait_exponential, stop_after_attempt

logger = structlog.get_logger(__name__)

# Prometheus metrics
MESSAGES_CONSUMED = Counter('kafka_messages_consumed_total', 'Total consumed messages', ['topic', 'partition'])
PROCESSING_TIME = Histogram('kafka_message_processing_seconds', 'Message processing time', ['topic'])
LAG_GAUGE = Gauge('kafka_consumer_lag', 'Consumer lag', ['topic', 'partition'])
THROUGHPUT_GAUGE = Gauge('kafka_throughput_messages_per_second', 'Messages per second', ['topic'])
ERROR_COUNTER = Counter('kafka_consumer_errors_total', 'Consumer errors', ['error_type'])

@dataclass
class KafkaConfig:
    """Configuração otimizada para Kafka industrial"""
    
    # Connection Settings
    bootstrap_servers: List[str] = field(default_factory=lambda: ['localhost:9092'])
    security_protocol: str = 'PLAINTEXT'
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    
    # Consumer Configuration - OPTIMIZED FOR THROUGHPUT
    group_id: str = 'crmbet-etl-consumer'
    client_id: str = 'crmbet-etl'
    auto_offset_reset: str = 'latest'  # 'earliest' for full reprocessing
    enable_auto_commit: bool = False   # Manual commit for reliability
    auto_commit_interval_ms: int = 1000
    
    # Performance Optimization
    fetch_min_bytes: int = 50000       # 50KB minimum fetch
    fetch_max_wait_ms: int = 500       # 500ms max wait
    max_partition_fetch_bytes: int = 1048576 * 10  # 10MB per partition
    receive_buffer_bytes: int = 65536 * 64  # 4MB receive buffer
    send_buffer_bytes: int = 131072 * 4     # 512KB send buffer
    
    # Session Management
    session_timeout_ms: int = 30000    # 30s session timeout
    heartbeat_interval_ms: int = 3000  # 3s heartbeat
    max_poll_interval_ms: int = 300000 # 5min max poll interval
    max_poll_records: int = 5000       # Max records per poll
    
    # Retry & Reliability
    retry_backoff_ms: int = 100
    reconnect_backoff_ms: int = 50
    reconnect_backoff_max_ms: int = 1000
    
    # Batch Processing
    batch_size: int = 10000           # Records per batch
    batch_timeout_ms: int = 5000      # 5s batch timeout
    max_memory_usage_mb: int = 2048   # 2GB max memory
    
    # Compression
    compression_type: str = 'lz4'     # Fast compression
    
    # Dead Letter Queue
    enable_dlq: bool = True
    dlq_topic: str = 'crmbet-etl-dlq'
    max_retries: int = 3
    
    # Topics Configuration
    topics: List[str] = field(default_factory=lambda: [
        'user-events',
        'transactions', 
        'game-sessions',
        'deposits-withdrawals'
    ])
    
    # Processing Configuration
    enable_preprocessing: bool = True
    enable_deduplication: bool = True
    dedup_window_minutes: int = 5
    enable_schema_validation: bool = True

@dataclass
class MessageBatch:
    """Batch de mensagens para processamento eficiente"""
    messages: List[Dict[str, Any]]
    topic: str
    partition: int
    offset_start: int
    offset_end: int
    timestamp_start: datetime
    timestamp_end: datetime
    total_bytes: int = 0
    
    @property
    def size(self) -> int:
        return len(self.messages)
    
    @property
    def timespan_seconds(self) -> float:
        return (self.timestamp_end - self.timestamp_start).total_seconds()

@dataclass
class ConsumerMetrics:
    """Métricas do consumer Kafka"""
    messages_consumed: int = 0
    bytes_consumed: int = 0
    batches_processed: int = 0
    errors_count: int = 0
    last_commit_time: Optional[datetime] = None
    current_lag: Dict[str, int] = field(default_factory=dict)
    throughput_msg_per_sec: float = 0.0
    avg_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

class KafkaStreamConsumer:
    """
    Consumer Kafka industrial para streaming ETL
    Otimizado para processamento de TB+/hora com latência <5min
    """
    
    def __init__(self, config: KafkaConfig, message_handler: Optional[Callable] = None):
        self.config = config
        self.message_handler = message_handler
        self.logger = logger.bind(component="KafkaStreamConsumer")
        
        # Consumer instances
        self.consumer: Optional[KafkaConsumer] = None
        self.confluent_consumer: Optional[ConfluentConsumer] = None
        
        # State management
        self.running = False
        self.metrics = ConsumerMetrics()
        self._message_buffer = deque(maxlen=self.config.batch_size * 2)
        self._last_metrics_update = time.time()
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=KafkaError
        )
        
        # Processing optimization
        self._dedup_cache = {}  # For deduplication
        self._schema_cache = {}  # For schema validation
        
        self.logger.info("KafkaStreamConsumer inicializado", config=self.config.__dict__)
    
    async def start_consuming(self) -> None:
        """Inicia consumer com otimizações industriais"""
        
        try:
            self.logger.info("Iniciando consumer Kafka")
            
            # Inicializa consumer
            self._initialize_consumer()
            
            # Verifica tópicos existem
            await self._verify_topics()
            
            # Inicia loop de consumo
            self.running = True
            
            # Usa async para não bloquear
            await asyncio.gather(
                self._consume_loop(),
                self._metrics_loop(),
                self._cleanup_loop()
            )
            
        except Exception as e:
            self.logger.error("Erro ao iniciar consumer", error=str(e))
            raise
    
    def _initialize_consumer(self) -> None:
        """Inicializa consumer com configuração otimizada"""
        
        consumer_config = {
            'bootstrap_servers': self.config.bootstrap_servers,
            'group_id': self.config.group_id,
            'client_id': self.config.client_id,
            'auto_offset_reset': self.config.auto_offset_reset,
            'enable_auto_commit': self.config.enable_auto_commit,
            'auto_commit_interval_ms': self.config.auto_commit_interval_ms,
            'fetch_min_bytes': self.config.fetch_min_bytes,
            'fetch_max_wait_ms': self.config.fetch_max_wait_ms,
            'max_partition_fetch_bytes': self.config.max_partition_fetch_bytes,
            'receive_buffer_bytes': self.config.receive_buffer_bytes,
            'send_buffer_bytes': self.config.send_buffer_bytes,
            'session_timeout_ms': self.config.session_timeout_ms,
            'heartbeat_interval_ms': self.config.heartbeat_interval_ms,
            'max_poll_interval_ms': self.config.max_poll_interval_ms,
            'max_poll_records': self.config.max_poll_records,
            'retry_backoff_ms': self.config.retry_backoff_ms,
            'reconnect_backoff_ms': self.config.reconnect_backoff_ms,
            'reconnect_backoff_max_ms': self.config.reconnect_backoff_max_ms,
            
            # Serializers para performance
            'value_deserializer': self._deserialize_message,
            'key_deserializer': lambda x: x.decode('utf-8') if x else None,
        }
        
        # Adiciona configurações de segurança se fornecidas
        if self.config.security_protocol != 'PLAINTEXT':
            consumer_config['security_protocol'] = self.config.security_protocol
            
        if self.config.sasl_mechanism:
            consumer_config['sasl_mechanism'] = self.config.sasl_mechanism
            consumer_config['sasl_plain_username'] = self.config.sasl_username
            consumer_config['sasl_plain_password'] = self.config.sasl_password
        
        # Cria consumer
        self.consumer = KafkaConsumer(**consumer_config)
        self.consumer.subscribe(self.config.topics)
        
        self.logger.info("Consumer Kafka inicializado", topics=self.config.topics)
    
    def _deserialize_message(self, raw_value: bytes) -> Dict[str, Any]:
        """Deserializa mensagem com otimizações de performance"""
        
        if not raw_value:
            return {}
        
        try:
            # Tenta decompressão LZ4 primeiro
            if raw_value.startswith(b'LZ4'):
                decompressed = lz4.frame.decompress(raw_value[3:])
                return orjson.loads(decompressed)
            
            # Tenta MessagePack
            if raw_value[0:1] in (b'\x80', b'\x90', b'\xa0'):
                return msgpack.unpackb(raw_value, raw=False)
            
            # Fallback para JSON
            return orjson.loads(raw_value)
            
        except Exception as e:
            self.logger.warning("Erro na deserialização", error=str(e))
            ERROR_COUNTER.labels(error_type='deserialization').inc()
            return {'raw_data': raw_value.decode('utf-8', errors='ignore')}
    
    async def _verify_topics(self) -> None:
        """Verifica se tópicos existem e cria se necessário"""
        
        try:
            # Usa admin client para verificar tópicos
            admin_config = {
                'bootstrap_servers': ','.join(self.config.bootstrap_servers)
            }
            
            admin_client = AdminClient(admin_config)
            metadata = admin_client.list_topics(timeout=10)
            
            existing_topics = set(metadata.topics.keys())
            required_topics = set(self.config.topics)
            
            missing_topics = required_topics - existing_topics
            
            if missing_topics:
                self.logger.warning("Tópicos faltando", topics=list(missing_topics))
                # Em produção, tópicos devem ser criados externamente
            
            self.logger.info("Verificação de tópicos concluída", 
                           existing=list(existing_topics), 
                           required=list(required_topics))
            
        except Exception as e:
            self.logger.error("Erro ao verificar tópicos", error=str(e))
    
    async def _consume_loop(self) -> None:
        """Loop principal de consumo otimizado"""
        
        batch_buffer = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Poll mensagens com timeout
                message_batch = self.consumer.poll(timeout_ms=1000, max_records=self.config.max_poll_records)
                
                if not message_batch:
                    # Processa batch parcial se timeout atingido
                    if batch_buffer and (time.time() - last_batch_time) > (self.config.batch_timeout_ms / 1000):
                        await self._process_batch(batch_buffer)
                        batch_buffer = []
                        last_batch_time = time.time()
                    continue
                
                # Processa mensagens por partição
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        if message.error():
                            self.logger.error("Erro na mensagem", error=message.error())
                            ERROR_COUNTER.labels(error_type='message_error').inc()
                            continue
                        
                        # Adiciona mensagem ao buffer
                        processed_msg = await self._preprocess_message(message)
                        if processed_msg:  # None se rejeitada por qualidade
                            batch_buffer.append(processed_msg)
                        
                        # Atualiza métricas
                        MESSAGES_CONSUMED.labels(
                            topic=message.topic(), 
                            partition=message.partition()
                        ).inc()
                        
                        self.metrics.messages_consumed += 1
                        self.metrics.bytes_consumed += len(message.value() or b'')
                
                # Processa batch quando atingir tamanho ou timeout
                current_time = time.time()
                
                if (len(batch_buffer) >= self.config.batch_size or 
                    (batch_buffer and (current_time - last_batch_time) > (self.config.batch_timeout_ms / 1000))):
                    
                    await self._process_batch(batch_buffer)
                    batch_buffer = []
                    last_batch_time = current_time
                
                # Controle de memória
                await self._memory_management()
                
            except Exception as e:
                self.logger.error("Erro no loop de consumo", error=str(e))
                ERROR_COUNTER.labels(error_type='consume_loop').inc()
                await asyncio.sleep(1)  # Backoff em caso de erro
    
    async def _preprocess_message(self, message) -> Optional[Dict[str, Any]]:
        """Pré-processa mensagem com validações"""
        
        try:
            # Extrai dados básicos
            msg_data = {
                'topic': message.topic(),
                'partition': message.partition(),
                'offset': message.offset(),
                'timestamp': datetime.fromtimestamp(message.timestamp() / 1000),
                'key': message.key(),
                'value': message.value(),
                'headers': dict(message.headers() or [])
            }
            
            # Deduplicação se habilitada
            if self.config.enable_deduplication:
                msg_id = self._generate_message_id(msg_data)
                if self._is_duplicate(msg_id):
                    self.logger.debug("Mensagem duplicada ignorada", msg_id=msg_id)
                    return None
                self._mark_as_processed(msg_id)
            
            # Validação de schema se habilitada
            if self.config.enable_schema_validation:
                if not self._validate_message_schema(msg_data):
                    self.logger.warning("Mensagem rejeitada por schema inválido")
                    return None
            
            return msg_data
            
        except Exception as e:
            self.logger.error("Erro no pré-processamento", error=str(e))
            return None
    
    def _generate_message_id(self, msg_data: Dict[str, Any]) -> str:
        """Gera ID único para deduplicação"""
        
        # Usa combinação de topic, partition, offset para ID único
        if msg_data.get('key'):
            return f"{msg_data['topic']}:{msg_data['partition']}:{msg_data['key']}"
        else:
            return f"{msg_data['topic']}:{msg_data['partition']}:{msg_data['offset']}"
    
    def _is_duplicate(self, msg_id: str) -> bool:
        """Verifica se mensagem é duplicata"""
        
        current_time = time.time()
        
        # Remove entradas antigas do cache
        cutoff_time = current_time - (self.config.dedup_window_minutes * 60)
        self._dedup_cache = {
            k: v for k, v in self._dedup_cache.items() 
            if v > cutoff_time
        }
        
        return msg_id in self._dedup_cache
    
    def _mark_as_processed(self, msg_id: str) -> None:
        """Marca mensagem como processada"""
        self._dedup_cache[msg_id] = time.time()
    
    def _validate_message_schema(self, msg_data: Dict[str, Any]) -> bool:
        """Valida schema da mensagem"""
        
        # Validações básicas
        if not msg_data.get('value'):
            return False
        
        # Validações específicas por tópico
        topic = msg_data['topic']
        
        try:
            if topic == 'user-events':
                return self._validate_user_event_schema(msg_data['value'])
            elif topic == 'transactions':
                return self._validate_transaction_schema(msg_data['value'])
            elif topic == 'game-sessions':
                return self._validate_game_session_schema(msg_data['value'])
            elif topic == 'deposits-withdrawals':
                return self._validate_financial_schema(msg_data['value'])
            
            return True  # Default: aceita se não tem validação específica
            
        except Exception as e:
            self.logger.error("Erro na validação de schema", topic=topic, error=str(e))
            return False
    
    def _validate_user_event_schema(self, value: Dict[str, Any]) -> bool:
        """Valida schema de eventos de usuário"""
        required_fields = ['user_id', 'event_type', 'timestamp']
        return all(field in value for field in required_fields)
    
    def _validate_transaction_schema(self, value: Dict[str, Any]) -> bool:
        """Valida schema de transações"""
        required_fields = ['user_id', 'amount', 'currency', 'transaction_type', 'timestamp']
        return all(field in value for field in required_fields)
    
    def _validate_game_session_schema(self, value: Dict[str, Any]) -> bool:
        """Valida schema de sessões de jogo"""
        required_fields = ['user_id', 'game_type', 'session_start', 'session_duration']
        return all(field in value for field in required_fields)
    
    def _validate_financial_schema(self, value: Dict[str, Any]) -> bool:
        """Valida schema de operações financeiras"""
        required_fields = ['user_id', 'amount', 'operation_type', 'timestamp', 'status']
        return all(field in value for field in required_fields)
    
    @circuit_breaker
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Processa batch de mensagens com circuit breaker"""
        
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            self.logger.info("Processando batch", size=len(batch))
            
            # Converte para DataFrame para processamento eficiente
            df = pd.DataFrame(batch)
            
            # Aplica handler personalizado se fornecido
            if self.message_handler:
                processed_df = await self._apply_message_handler(df)
            else:
                processed_df = df
            
            # Atualiza métricas
            processing_time = time.time() - start_time
            self.metrics.batches_processed += 1
            self.metrics.avg_processing_time_ms = (
                (self.metrics.avg_processing_time_ms * (self.metrics.batches_processed - 1) + 
                 processing_time * 1000) / self.metrics.batches_processed
            )
            
            # Prometheus metrics
            PROCESSING_TIME.labels(topic='batch').observe(processing_time)
            
            # Commit offsets após processamento bem-sucedido
            if not self.config.enable_auto_commit:
                await self._commit_offsets(batch)
            
            self.logger.info(
                "Batch processado com sucesso",
                size=len(batch),
                processing_time_ms=processing_time * 1000
            )
            
        except Exception as e:
            self.logger.error("Erro no processamento do batch", error=str(e))
            ERROR_COUNTER.labels(error_type='batch_processing').inc()
            
            # Envia para DLQ se habilitado
            if self.config.enable_dlq:
                await self._send_to_dlq(batch, str(e))
            
            raise
    
    async def _apply_message_handler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica handler personalizado de mensagens"""
        
        try:
            if asyncio.iscoroutinefunction(self.message_handler):
                return await self.message_handler(df)
            else:
                # Execute in thread pool for CPU-bound operations
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(executor, self.message_handler, df)
                    
        except Exception as e:
            self.logger.error("Erro no handler de mensagens", error=str(e))
            raise
    
    async def _commit_offsets(self, batch: List[Dict[str, Any]]) -> None:
        """Commit offsets de forma otimizada"""
        
        try:
            # Agrupa por tópico/partição
            partitions_offsets = {}
            
            for msg in batch:
                key = (msg['topic'], msg['partition'])
                if key not in partitions_offsets:
                    partitions_offsets[key] = msg['offset']
                else:
                    partitions_offsets[key] = max(partitions_offsets[key], msg['offset'])
            
            # Cria TopicPartitions para commit
            partitions_to_commit = {}
            for (topic, partition), offset in partitions_offsets.items():
                tp = TopicPartition(topic, partition)
                partitions_to_commit[tp] = offset + 1  # Próximo offset
            
            # Executa commit
            self.consumer.commit(partitions_to_commit)
            self.metrics.last_commit_time = datetime.now()
            
            self.logger.debug("Offsets commitados", partitions=len(partitions_to_commit))
            
        except CommitFailedError as e:
            self.logger.error("Falha no commit de offsets", error=str(e))
            ERROR_COUNTER.labels(error_type='commit_failed').inc()
            raise
    
    async def _send_to_dlq(self, batch: List[Dict[str, Any]], error: str) -> None:
        """Envia mensagens falhadas para Dead Letter Queue"""
        
        try:
            # Implementar producer para DLQ aqui
            # Por simplicidade, apenas logamos por enquanto
            self.logger.warning(
                "Enviando para DLQ",
                batch_size=len(batch),
                error=error,
                dlq_topic=self.config.dlq_topic
            )
            
        except Exception as e:
            self.logger.error("Erro ao enviar para DLQ", error=str(e))
    
    async def _memory_management(self) -> None:
        """Gerenciamento de memória durante consumo"""
        
        # Verifica uso de memória
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.metrics.memory_usage_mb = memory_mb
        
        # Se exceder limite, força limpeza
        if memory_mb > self.config.max_memory_usage_mb:
            self.logger.warning("Uso alto de memória, executando limpeza", memory_mb=memory_mb)
            
            # Limpa caches
            self._dedup_cache.clear()
            self._schema_cache.clear()
            
            # Força garbage collection
            import gc
            gc.collect()
    
    async def _metrics_loop(self) -> None:
        """Loop de atualização de métricas"""
        
        while self.running:
            try:
                # Atualiza métricas de throughput
                current_time = time.time()
                time_delta = current_time - self._last_metrics_update
                
                if time_delta >= 1.0:  # Update every second
                    self.metrics.throughput_msg_per_sec = (
                        self.metrics.messages_consumed / time_delta
                    )
                    
                    # Reset counter
                    self.metrics.messages_consumed = 0
                    self._last_metrics_update = current_time
                    
                    # Update Prometheus metrics
                    THROUGHPUT_GAUGE.labels(topic='all').set(self.metrics.throughput_msg_per_sec)
                
                # Atualiza lag do consumer
                await self._update_consumer_lag()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error("Erro no loop de métricas", error=str(e))
                await asyncio.sleep(10)
    
    async def _update_consumer_lag(self) -> None:
        """Atualiza lag do consumer"""
        
        try:
            # Obtém partições atribuídas
            assignment = self.consumer.assignment()
            
            for tp in assignment:
                # Posição atual do consumer
                position = self.consumer.position(tp)
                
                # High water mark (último offset)
                high_water_mark = self.consumer.get_partition_metadata(tp.topic)[tp.partition].high_water_mark
                
                # Calcula lag
                lag = high_water_mark - position
                self.metrics.current_lag[f"{tp.topic}:{tp.partition}"] = lag
                
                # Update Prometheus
                LAG_GAUGE.labels(topic=tp.topic, partition=tp.partition).set(lag)
                
        except Exception as e:
            self.logger.error("Erro ao calcular lag", error=str(e))
    
    async def _cleanup_loop(self) -> None:
        """Loop de limpeza periódica"""
        
        while self.running:
            try:
                # Executa limpeza a cada 5 minutos
                await asyncio.sleep(300)
                
                # Limpa cache de deduplicação
                current_time = time.time()
                cutoff_time = current_time - (self.config.dedup_window_minutes * 60)
                
                self._dedup_cache = {
                    k: v for k, v in self._dedup_cache.items() 
                    if v > cutoff_time
                }
                
                self.logger.debug("Limpeza de cache executada", cache_size=len(self._dedup_cache))
                
            except Exception as e:
                self.logger.error("Erro na limpeza", error=str(e))
    
    async def stop_consuming(self) -> None:
        """Para consumer de forma limpa"""
        
        try:
            self.logger.info("Parando consumer Kafka")
            
            self.running = False
            
            # Processa mensagens restantes no buffer
            if self._message_buffer:
                await self._process_batch(list(self._message_buffer))
                self._message_buffer.clear()
            
            # Fecha consumer
            if self.consumer:
                self.consumer.close()
                self.consumer = None
            
            self.logger.info("Consumer Kafka parado com sucesso")
            
        except Exception as e:
            self.logger.error("Erro ao parar consumer", error=str(e))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do consumer"""
        
        return {
            'messages_consumed': self.metrics.messages_consumed,
            'bytes_consumed': self.metrics.bytes_consumed,
            'batches_processed': self.metrics.batches_processed,
            'errors_count': self.metrics.errors_count,
            'throughput_msg_per_sec': self.metrics.throughput_msg_per_sec,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'current_lag': self.metrics.current_lag,
            'last_commit_time': self.metrics.last_commit_time.isoformat() if self.metrics.last_commit_time else None,
            'dedup_cache_size': len(self._dedup_cache),
            'running': self.running
        }

# Context manager para uso fácil
class KafkaConsumerContext:
    """Context manager para consumer Kafka"""
    
    def __init__(self, config: KafkaConfig, message_handler: Optional[Callable] = None):
        self.consumer = KafkaStreamConsumer(config, message_handler)
    
    async def __aenter__(self):
        return self.consumer
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.consumer.stop_consuming()

# Função utilitária para processamento simples
async def consume_kafka_stream(topics: List[str], 
                             message_handler: Callable,
                             config: Optional[KafkaConfig] = None) -> None:
    """
    Função utilitária para consumo de stream Kafka
    
    Args:
        topics: Lista de tópicos
        message_handler: Função para processar mensagens
        config: Configuração Kafka
    """
    
    if config is None:
        config = KafkaConfig()
        config.topics = topics
    
    async with KafkaConsumerContext(config, message_handler) as consumer:
        await consumer.start_consuming()