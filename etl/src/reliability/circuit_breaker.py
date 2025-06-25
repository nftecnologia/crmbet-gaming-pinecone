"""
🚨 Industrial Circuit Breaker - MAXIMUM FAULT TOLERANCE
Advanced circuit breaker with adaptive thresholds and self-healing capabilities

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import structlog
from collections import deque, defaultdict
import inspect

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge, Enum as PrometheusEnum
import psutil

# Exception handling
import traceback
from contextlib import asynccontextmanager, contextmanager

logger = structlog.get_logger(__name__)

# Prometheus metrics
CIRCUIT_BREAKER_STATE = PrometheusEnum(
    'circuit_breaker_state', 
    'Circuit breaker state', 
    ['service'], 
    states=['closed', 'open', 'half_open']
)
CIRCUIT_BREAKER_FAILURES = Counter(
    'circuit_breaker_failures_total', 
    'Total circuit breaker failures', 
    ['service', 'error_type']
)
CIRCUIT_BREAKER_CALLS = Counter(
    'circuit_breaker_calls_total', 
    'Total circuit breaker calls', 
    ['service', 'result']
)
CIRCUIT_BREAKER_LATENCY = Histogram(
    'circuit_breaker_latency_seconds', 
    'Circuit breaker call latency', 
    ['service']
)

class CircuitState(Enum):
    """Estados do circuit breaker"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Blocking calls
    HALF_OPEN = "half_open"    # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuração avançada do circuit breaker"""
    
    # Thresholds básicos
    failure_threshold: int = 5          # Número de falhas para abrir
    recovery_timeout: int = 60          # Segundos para tentar recovery
    success_threshold: int = 3          # Sucessos para fechar
    
    # Adaptive thresholds
    enable_adaptive: bool = True
    adaptive_window_size: int = 100     # Tamanho da janela para análise
    adaptive_factor: float = 1.5        # Fator de adaptação
    
    # Rate limiting
    max_calls_per_minute: int = 1000    # Máximo de calls por minuto
    burst_threshold: int = 50           # Threshold para rajadas
    
    # Error analysis
    error_rate_threshold: float = 0.5   # 50% error rate
    latency_threshold_ms: float = 5000  # 5s latency threshold
    
    # Recovery testing
    half_open_max_calls: int = 5        # Máximo calls em half-open
    
    # Monitoring
    enable_detailed_metrics: bool = True
    alert_on_state_change: bool = True
    
    # Exception handling
    excluded_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        KeyboardInterrupt, SystemExit
    ])
    retriable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, OSError
    ])

@dataclass
class CallResult:
    """Resultado de uma chamada"""
    success: bool
    latency_ms: float
    timestamp: datetime
    exception: Optional[Exception] = None
    error_type: Optional[str] = None

@dataclass
class CircuitMetrics:
    """Métricas do circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state_changes: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

class IndustrialCircuitBreaker:
    """
    Circuit Breaker industrial com capacidades avançadas:
    - Adaptive thresholds baseados em histórico
    - Rate limiting integrado
    - Análise de tipos de erro
    - Self-healing automático
    - Métricas detalhadas
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logger.bind(component="CircuitBreaker", name=name)
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()
        
        # Metrics and history
        self.metrics = CircuitMetrics()
        self.call_history = deque(maxlen=self.config.adaptive_window_size)
        self.error_patterns = defaultdict(int)
        
        # Adaptive thresholds
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.adaptive_latency_threshold = self.config.latency_threshold_ms
        
        # Rate limiting
        self.call_timestamps = deque(maxlen=self.config.max_calls_per_minute)
        
        # Threading
        self._lock = threading.RLock()
        
        # Background tasks
        self._monitoring_task = None
        self._recovery_task = None
        
        self.logger.info("Circuit breaker inicializado", config=self.config.__dict__)
        
        # Start background monitoring
        if self.config.enable_detailed_metrics:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Inicia monitoramento em background"""
        try:
            loop = asyncio.get_event_loop()
            self._monitoring_task = loop.create_task(self._monitoring_loop())
        except RuntimeError:
            # No event loop, skip background monitoring
            pass
    
    async def _monitoring_loop(self):
        """Loop de monitoramento contínuo"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update adaptive thresholds
                self._update_adaptive_thresholds()
                
                # Analyze error patterns
                self._analyze_error_patterns()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics()
                
                # Check for auto-recovery
                await self._check_auto_recovery()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Erro no loop de monitoramento", error=str(e))
    
    def _update_adaptive_thresholds(self):
        """Atualiza thresholds adaptativos baseado no histórico"""
        
        if not self.config.enable_adaptive or len(self.call_history) < 20:
            return
        
        with self._lock:
            # Análise de latência
            latencies = [call.latency_ms for call in self.call_history if call.success]
            if latencies:
                avg_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                
                # Ajusta threshold de latência
                new_latency_threshold = avg_latency + (std_latency * self.config.adaptive_factor)
                self.adaptive_latency_threshold = min(
                    new_latency_threshold,
                    self.config.latency_threshold_ms * 2  # Limite máximo
                )
            
            # Análise de failure rate
            recent_failures = sum(1 for call in self.call_history if not call.success)
            failure_rate = recent_failures / len(self.call_history)
            
            if failure_rate > 0.1:  # Se > 10% failures
                # Reduz threshold de failure
                self.adaptive_failure_threshold = max(
                    int(self.config.failure_threshold * 0.7),
                    2  # Mínimo de 2
                )
            else:
                # Aumenta threshold gradualmente
                self.adaptive_failure_threshold = min(
                    int(self.adaptive_failure_threshold * 1.1),
                    self.config.failure_threshold * 2
                )
            
            self.logger.debug(
                "Thresholds adaptativos atualizados",
                failure_threshold=self.adaptive_failure_threshold,
                latency_threshold=self.adaptive_latency_threshold
            )
    
    def _analyze_error_patterns(self):
        """Analisa padrões de erro para insights"""
        
        if len(self.call_history) < 10:
            return
        
        # Conta tipos de erro
        error_counts = defaultdict(int)
        for call in self.call_history:
            if not call.success and call.error_type:
                error_counts[call.error_type] += 1
        
        # Identifica padrões
        total_errors = sum(error_counts.values())
        if total_errors > 0:
            dominant_error = max(error_counts.items(), key=lambda x: x[1])
            
            if dominant_error[1] / total_errors > 0.7:  # 70% do mesmo tipo
                self.logger.warning(
                    "Padrão de erro dominante detectado",
                    error_type=dominant_error[0],
                    percentage=dominant_error[1] / total_errors
                )
    
    def _update_prometheus_metrics(self):
        """Atualiza métricas Prometheus"""
        
        CIRCUIT_BREAKER_STATE.labels(service=self.name).state(self.state.value)
        CIRCUIT_BREAKER_CALLS.labels(service=self.name, result='success')._value._value = self.metrics.successful_calls
        CIRCUIT_BREAKER_CALLS.labels(service=self.name, result='failure')._value._value = self.metrics.failed_calls
    
    async def _check_auto_recovery(self):
        """Verifica se pode fazer auto-recovery"""
        
        if self.state == CircuitState.OPEN:
            time_since_last_failure = time.time() - (self.last_failure_time or 0)
            
            if time_since_last_failure > self.config.recovery_timeout:
                self.logger.info("Tentando auto-recovery")
                self._transition_to_half_open()
    
    def _is_rate_limited(self) -> bool:
        """Verifica se está rate limited"""
        
        current_time = time.time()
        
        # Remove timestamps antigos (> 1 minuto)
        cutoff_time = current_time - 60
        while self.call_timestamps and self.call_timestamps[0] < cutoff_time:
            self.call_timestamps.popleft()
        
        # Verifica limite
        if len(self.call_timestamps) >= self.config.max_calls_per_minute:
            return True
        
        # Verifica burst
        recent_calls = sum(1 for ts in self.call_timestamps if current_time - ts < 10)
        if recent_calls >= self.config.burst_threshold:
            return True
        
        return False
    
    def _should_exclude_exception(self, exception: Exception) -> bool:
        """Verifica se exceção deve ser excluída do circuit breaker"""
        
        for exc_type in self.config.excluded_exceptions:
            if isinstance(exception, exc_type):
                return True
        return False
    
    def _is_retriable_exception(self, exception: Exception) -> bool:
        """Verifica se exceção é retriable"""
        
        for exc_type in self.config.retriable_exceptions:
            if isinstance(exception, exc_type):
                return True
        return False
    
    def _record_call(self, result: CallResult):
        """Registra resultado de uma chamada"""
        
        with self._lock:
            # Atualiza métricas
            self.metrics.total_calls += 1
            
            if result.success:
                self.metrics.successful_calls += 1
                self.metrics.last_success_time = result.timestamp
                self.success_count += 1
                self.failure_count = 0  # Reset failure count on success
            else:
                self.metrics.failed_calls += 1
                self.metrics.last_failure_time = result.timestamp
                self.last_failure_time = time.time()
                self.failure_count += 1
                self.success_count = 0  # Reset success count on failure
                
                # Conta tipo de erro
                if result.error_type:
                    self.error_patterns[result.error_type] += 1
            
            # Atualiza latência média
            total_latency = self.metrics.avg_latency_ms * (self.metrics.total_calls - 1)
            self.metrics.avg_latency_ms = (total_latency + result.latency_ms) / self.metrics.total_calls
            
            # Atualiza error rate
            self.metrics.error_rate = self.metrics.failed_calls / self.metrics.total_calls
            
            # Adiciona ao histórico
            self.call_history.append(result)
            
            # Atualiza timestamp para rate limiting
            self.call_timestamps.append(time.time())
    
    def _transition_to_open(self):
        """Transição para estado OPEN"""
        
        with self._lock:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()
                self.metrics.state_changes += 1
                
                self.logger.warning(
                    "Circuit breaker ABERTO",
                    failure_count=self.failure_count,
                    error_rate=self.metrics.error_rate
                )
                
                if self.config.alert_on_state_change:
                    self._send_alert("Circuit breaker opened due to failures")
    
    def _transition_to_half_open(self):
        """Transição para estado HALF_OPEN"""
        
        with self._lock:
            if self.state == CircuitState.OPEN:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                self.metrics.state_changes += 1
                self.success_count = 0
                
                self.logger.info("Circuit breaker em HALF-OPEN - testando recovery")
    
    def _transition_to_closed(self):
        """Transição para estado CLOSED"""
        
        with self._lock:
            if self.state != CircuitState.CLOSED:
                self.state = CircuitState.CLOSED
                self.last_state_change = datetime.now()
                self.metrics.state_changes += 1
                self.failure_count = 0
                
                self.logger.info("Circuit breaker FECHADO - recovery bem-sucedido")
                
                if self.config.alert_on_state_change:
                    self._send_alert("Circuit breaker recovered and closed")
    
    def _send_alert(self, message: str):
        """Envia alerta (implementar integração específica)"""
        
        # Por enquanto apenas log
        self.logger.warning("ALERTA Circuit Breaker", message=message, service=self.name)
    
    def _should_allow_call(self) -> bool:
        """Verifica se deve permitir a chamada"""
        
        # Verifica rate limiting
        if self._is_rate_limited():
            return False
        
        # Estados do circuit breaker
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Permite apenas número limitado de calls
            return self.success_count < self.config.half_open_max_calls
        
        return False
    
    def _evaluate_state_after_call(self, result: CallResult):
        """Avalia se deve mudar estado após chamada"""
        
        with self._lock:
            if self.state == CircuitState.CLOSED:
                # Verifica se deve abrir
                if (not result.success and 
                    self.failure_count >= self.adaptive_failure_threshold):
                    self._transition_to_open()
                elif (result.latency_ms > self.adaptive_latency_threshold and
                      result.success):
                    # Considera alta latência como falha parcial
                    self.logger.warning(
                        "Alta latência detectada",
                        latency_ms=result.latency_ms,
                        threshold=self.adaptive_latency_threshold
                    )
            
            elif self.state == CircuitState.HALF_OPEN:
                if result.success:
                    if self.success_count >= self.config.success_threshold:
                        self._transition_to_closed()
                else:
                    # Falha em half-open volta para open
                    self._transition_to_open()
    
    @contextmanager
    def _call_context(self):
        """Context manager para chamadas síncronas"""
        
        if not self._should_allow_call():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        start_time = time.time()
        exception = None
        
        try:
            yield
        except Exception as e:
            exception = e
            raise
        finally:
            # Registra resultado
            latency_ms = (time.time() - start_time) * 1000
            
            result = CallResult(
                success=exception is None,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                exception=exception,
                error_type=type(exception).__name__ if exception else None
            )
            
            # Pula exceções excluídas
            if exception and self._should_exclude_exception(exception):
                return
            
            self._record_call(result)
            self._evaluate_state_after_call(result)
            
            # Prometheus metrics
            CIRCUIT_BREAKER_LATENCY.labels(service=self.name).observe(latency_ms / 1000)
            
            if exception:
                CIRCUIT_BREAKER_FAILURES.labels(
                    service=self.name, 
                    error_type=type(exception).__name__
                ).inc()
    
    @asynccontextmanager
    async def _async_call_context(self):
        """Context manager para chamadas assíncronas"""
        
        if not self._should_allow_call():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        start_time = time.time()
        exception = None
        
        try:
            yield
        except Exception as e:
            exception = e
            raise
        finally:
            # Registra resultado
            latency_ms = (time.time() - start_time) * 1000
            
            result = CallResult(
                success=exception is None,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                exception=exception,
                error_type=type(exception).__name__ if exception else None
            )
            
            # Pula exceções excluídas
            if exception and self._should_exclude_exception(exception):
                return
            
            self._record_call(result)
            self._evaluate_state_after_call(result)
            
            # Prometheus metrics
            CIRCUIT_BREAKER_LATENCY.labels(service=self.name).observe(latency_ms / 1000)
            
            if exception:
                CIRCUIT_BREAKER_FAILURES.labels(
                    service=self.name, 
                    error_type=type(exception).__name__
                ).inc()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Executa função com circuit breaker (síncrono)"""
        
        with self._call_context():
            return func(*args, **kwargs)
    
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """Executa função com circuit breaker (assíncrono)"""
        
        async with self._async_call_context():
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator para circuit breaker"""
        
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper
    
    def async_decorator(self, func: Callable) -> Callable:
        """Decorator para funções assíncronas"""
        
        async def wrapper(*args, **kwargs):
            return await self.acall(func, *args, **kwargs)
        
        return wrapper
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do circuit breaker"""
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'adaptive_thresholds': {
                'failure_threshold': self.adaptive_failure_threshold,
                'latency_threshold_ms': self.adaptive_latency_threshold
            },
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'success_rate': self.metrics.success_rate,
                'error_rate': self.metrics.error_rate,
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'state_changes': self.metrics.state_changes
            },
            'error_patterns': dict(self.error_patterns),
            'last_state_change': self.last_state_change.isoformat(),
            'config': self.config.__dict__
        }
    
    def reset(self):
        """Reset manual do circuit breaker"""
        
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.call_history.clear()
            self.error_patterns.clear()
            
            self.logger.info("Circuit breaker resetado manualmente")
    
    def force_open(self):
        """Força abertura do circuit breaker"""
        
        with self._lock:
            self._transition_to_open()
            self.logger.warning("Circuit breaker forçado para OPEN")
    
    def force_close(self):
        """Força fechamento do circuit breaker"""
        
        with self._lock:
            self._transition_to_closed()
            self.logger.info("Circuit breaker forçado para CLOSED")

class CircuitBreakerOpenError(Exception):
    """Exceção quando circuit breaker está aberto"""
    pass

# Factory para criar circuit breakers configurados
class CircuitBreakerFactory:
    """Factory para circuit breakers com configurações pré-definidas"""
    
    _instances: Dict[str, IndustrialCircuitBreaker] = {}
    
    @classmethod
    def get_database_circuit_breaker(cls, name: str = "database") -> IndustrialCircuitBreaker:
        """Circuit breaker otimizado para operações de banco"""
        
        if name not in cls._instances:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                latency_threshold_ms=3000,  # 3s para DB
                max_calls_per_minute=500
            )
            cls._instances[name] = IndustrialCircuitBreaker(name, config)
        
        return cls._instances[name]
    
    @classmethod
    def get_api_circuit_breaker(cls, name: str = "api") -> IndustrialCircuitBreaker:
        """Circuit breaker otimizado para APIs externas"""
        
        if name not in cls._instances:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                latency_threshold_ms=10000,  # 10s para APIs
                max_calls_per_minute=1000
            )
            cls._instances[name] = IndustrialCircuitBreaker(name, config)
        
        return cls._instances[name]
    
    @classmethod
    def get_file_circuit_breaker(cls, name: str = "file") -> IndustrialCircuitBreaker:
        """Circuit breaker otimizado para operações de arquivo"""
        
        if name not in cls._instances:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=15,
                latency_threshold_ms=5000,  # 5s para files
                max_calls_per_minute=2000
            )
            cls._instances[name] = IndustrialCircuitBreaker(name, config)
        
        return cls._instances[name]

# Decorators convenientes
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator para aplicar circuit breaker"""
    
    cb = IndustrialCircuitBreaker(name, config)
    return cb

def async_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator para circuit breaker assíncrono"""
    
    cb = IndustrialCircuitBreaker(name, config)
    return cb.async_decorator