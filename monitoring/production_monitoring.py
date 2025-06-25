#!/usr/bin/env python3
"""
🚀 PRODUCTION MONITORING SYSTEM - Ultra-Robust Observability
Sistema completo de monitoramento enterprise para escala massiva

Author: Agente Monitoring - ULTRATHINK
Created: 2025-06-25
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import aioredis
import psutil
import structlog
from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry, 
    multiprocess, generate_latest, CONTENT_TYPE_LATEST
)
from dataclasses import dataclass, asdict
import subprocess
import requests

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@dataclass
class SystemHealth:
    """Sistema de health check completo"""
    overall_status: str
    components: Dict[str, str]
    metrics: Dict[str, float]
    alerts: List[str]
    timestamp: str
    uptime_seconds: float

@dataclass
class PerformanceMetrics:
    """Métricas de performance detalhadas"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_connections: int
    response_time_ms: float
    throughput_rps: float
    error_rate: float

class PrometheusMetrics:
    """Sistema de métricas Prometheus enterprise"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        
        # Application metrics
        self.http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'], registry=self.registry)
        self.http_request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'], registry=self.registry)
        self.active_connections = Gauge('active_connections_total', 'Active connections', registry=self.registry)
        
        # Business metrics
        self.transactions_total = Counter('transactions_total', 'Total transactions', ['type', 'status'], registry=self.registry)
        self.ml_predictions_total = Counter('ml_predictions_total', 'Total ML predictions', ['model', 'outcome'], registry=self.registry)
        self.etl_records_processed = Counter('etl_records_processed_total', 'ETL records processed', ['job_type'], registry=self.registry)
        
        # Performance metrics
        self.database_query_duration = Histogram('database_query_duration_seconds', 'Database query duration', ['query_type'], registry=self.registry)
        self.cache_hit_rate = Gauge('cache_hit_rate_percent', 'Cache hit rate percentage', ['cache_type'], registry=self.registry)
        self.queue_size = Gauge('queue_size_total', 'Queue size', ['queue_name'], registry=self.registry)
        
        # Health metrics
        self.system_health_score = Gauge('system_health_score', 'Overall system health score (0-1)', registry=self.registry)
        self.component_health = Gauge('component_health_status', 'Component health status', ['component'], registry=self.registry)
        
        logger.info("Prometheus metrics initialized")

class UltraRobustMonitoring:
    """Sistema de monitoramento ultra-robusto para produção"""
    
    def __init__(self):
        self.logger = logger.bind(component="UltraRobustMonitoring")
        self.metrics = PrometheusMetrics()
        self.start_time = time.time()
        
        # Configuration
        self.config = {
            'check_interval': int(os.getenv('MONITORING_INTERVAL', 30)),
            'alert_threshold_cpu': float(os.getenv('ALERT_THRESHOLD_CPU', 80.0)),
            'alert_threshold_memory': float(os.getenv('ALERT_THRESHOLD_MEMORY', 85.0)),
            'alert_threshold_disk': float(os.getenv('ALERT_THRESHOLD_DISK', 90.0)),
            'alert_threshold_response_time': float(os.getenv('ALERT_THRESHOLD_RESPONSE_TIME', 1000.0)),
            'enable_alerts': os.getenv('ENABLE_ALERTS', 'true').lower() == 'true',
            'webhook_url': os.getenv('WEBHOOK_ALERT_URL'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'database_url': os.getenv('DATABASE_URL'),
            'backend_url': os.getenv('BACKEND_URL', 'http://localhost:3001'),
            'ml_url': os.getenv('ML_URL', 'http://localhost:8000'),
        }
        
        # Component URLs for health checks
        self.components = {
            'backend': f"{self.config['backend_url']}/health",
            'ml_pipeline': f"{self.config['ml_url']}/health",
            'database': None,  # Special handling
            'redis': None,     # Special handling
        }
        
        self.logger.info("Ultra-robust monitoring system initialized", config=self.config)
    
    async def start_monitoring(self):
        """Inicia sistema de monitoramento contínuo"""
        
        self.logger.info("🚀 Iniciando monitoramento ultra-robusto")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._system_metrics_collector()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._alert_processor()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user")
        except Exception as e:
            self.logger.error("Monitoring error", error=str(e))
        finally:
            for task in tasks:
                task.cancel()
    
    async def _system_metrics_collector(self):
        """Coleta métricas do sistema continuamente"""
        
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.cpu_usage.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.metrics.memory_usage.set(memory.percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.metrics.disk_usage.set(disk_percent)
                
                # Network metrics
                network = psutil.net_io_counters()
                
                # Active connections (if backend is running)
                try:
                    connections = len(psutil.net_connections())
                    self.metrics.active_connections.set(connections)
                except:
                    pass
                
                self.logger.debug("System metrics collected", 
                                cpu=cpu_percent, 
                                memory=memory.percent, 
                                disk=disk_percent)
                
                await asyncio.sleep(self.config['check_interval'])
                
            except Exception as e:
                self.logger.error("Error collecting system metrics", error=str(e))
                await asyncio.sleep(5)
    
    async def _health_checker(self):
        """Executa health checks dos componentes"""
        
        while True:
            try:
                health_data = await self.get_comprehensive_health()
                
                # Update health metrics
                overall_score = self._calculate_health_score(health_data)
                self.metrics.system_health_score.set(overall_score)
                
                # Update component health
                for component, status in health_data.components.items():
                    health_value = 1.0 if status == 'healthy' else 0.0
                    self.metrics.component_health.labels(component=component).set(health_value)
                
                # Check for alerts
                if health_data.alerts and self.config['enable_alerts']:
                    await self._send_alerts(health_data.alerts)
                
                self.logger.info("Health check completed", 
                               overall_status=health_data.overall_status,
                               score=overall_score,
                               alerts_count=len(health_data.alerts))
                
                await asyncio.sleep(self.config['check_interval'])
                
            except Exception as e:
                self.logger.error("Error in health checker", error=str(e))
                await asyncio.sleep(10)
    
    async def _performance_monitor(self):
        """Monitora performance da aplicação"""
        
        while True:
            try:
                perf_metrics = await self.get_performance_metrics()
                
                # Check performance thresholds
                alerts = []
                
                if perf_metrics.cpu_percent > self.config['alert_threshold_cpu']:
                    alerts.append(f"High CPU usage: {perf_metrics.cpu_percent:.1f}%")
                
                if perf_metrics.memory_percent > self.config['alert_threshold_memory']:
                    alerts.append(f"High memory usage: {perf_metrics.memory_percent:.1f}%")
                
                if perf_metrics.disk_percent > self.config['alert_threshold_disk']:
                    alerts.append(f"High disk usage: {perf_metrics.disk_percent:.1f}%")
                
                if perf_metrics.response_time_ms > self.config['alert_threshold_response_time']:
                    alerts.append(f"High response time: {perf_metrics.response_time_ms:.1f}ms")
                
                if alerts and self.config['enable_alerts']:
                    await self._send_alerts(alerts)
                
                self.logger.debug("Performance monitoring completed", 
                                performance=asdict(perf_metrics))
                
                await asyncio.sleep(self.config['check_interval'] * 2)  # Less frequent
                
            except Exception as e:
                self.logger.error("Error in performance monitor", error=str(e))
                await asyncio.sleep(15)
    
    async def _alert_processor(self):
        """Processa alertas e notificações"""
        
        while True:
            try:
                # This would typically connect to a queue for alerts
                # For now, we'll just log that the alert processor is running
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error("Error in alert processor", error=str(e))
                await asyncio.sleep(30)
    
    async def get_comprehensive_health(self) -> SystemHealth:
        """Obtém saúde completa do sistema"""
        
        components_health = {}
        alerts = []
        
        # Check each component
        for component, url in self.components.items():
            try:
                if component == 'database':
                    status = await self._check_database_health()
                elif component == 'redis':
                    status = await self._check_redis_health()
                else:
                    status = await self._check_http_health(url)
                
                components_health[component] = status
                
            except Exception as e:
                components_health[component] = 'unhealthy'
                alerts.append(f"{component} health check failed: {str(e)}")
        
        # Check system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        
        if cpu_percent > self.config['alert_threshold_cpu']:
            alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory_percent > self.config['alert_threshold_memory']:
            alerts.append(f"High memory usage: {memory_percent:.1f}%")
        
        if disk_percent > self.config['alert_threshold_disk']:
            alerts.append(f"High disk usage: {disk_percent:.1f}%")
        
        # Determine overall status
        unhealthy_components = [k for k, v in components_health.items() if v != 'healthy']
        
        if not unhealthy_components and not alerts:
            overall_status = 'healthy'
        elif len(unhealthy_components) <= 1 and len(alerts) <= 2:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        return SystemHealth(
            overall_status=overall_status,
            components=components_health,
            metrics={
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'uptime_seconds': time.time() - self.start_time
            },
            alerts=alerts,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=time.time() - self.start_time
        )
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Obtém métricas de performance detalhadas"""
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Active connections
        try:
            active_connections = len(psutil.net_connections())
        except:
            active_connections = 0
        
        # Response time (test backend)
        response_time_ms = await self._measure_response_time()
        
        # Calculate throughput and error rate (simplified)
        throughput_rps = 0.0  # Would be calculated from actual metrics
        error_rate = 0.0      # Would be calculated from actual metrics
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io=network_io,
            active_connections=active_connections,
            response_time_ms=response_time_ms,
            throughput_rps=throughput_rps,
            error_rate=error_rate
        )
    
    async def _check_http_health(self, url: str) -> str:
        """Verifica saúde de endpoint HTTP"""
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return 'healthy'
                    else:
                        return 'unhealthy'
        except:
            return 'unhealthy'
    
    async def _check_database_health(self) -> str:
        """Verifica saúde do banco de dados"""
        
        if not self.config['database_url']:
            return 'unknown'
        
        try:
            # Simple connection test
            import asyncpg
            conn = await asyncpg.connect(self.config['database_url'])
            await conn.execute('SELECT 1')
            await conn.close()
            return 'healthy'
        except Exception as e:
            self.logger.warning("Database health check failed", error=str(e))
            return 'unhealthy'
    
    async def _check_redis_health(self) -> str:
        """Verifica saúde do Redis"""
        
        try:
            redis = aioredis.from_url(self.config['redis_url'])
            await redis.ping()
            await redis.close()
            return 'healthy'
        except Exception as e:
            self.logger.warning("Redis health check failed", error=str(e))
            return 'unhealthy'
    
    async def _measure_response_time(self) -> float:
        """Mede tempo de resposta do backend"""
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.config['backend_url']}/health") as response:
                    await response.text()
            
            return (time.time() - start_time) * 1000  # Convert to ms
        except:
            return 9999.0  # Timeout or error
    
    def _calculate_health_score(self, health_data: SystemHealth) -> float:
        """Calcula score de saúde geral (0-1)"""
        
        score = 1.0
        
        # Penalize unhealthy components
        unhealthy_components = [k for k, v in health_data.components.items() if v != 'healthy']
        score -= len(unhealthy_components) * 0.2
        
        # Penalize high resource usage
        if health_data.metrics['cpu_percent'] > 80:
            score -= 0.1
        if health_data.metrics['memory_percent'] > 85:
            score -= 0.1
        if health_data.metrics['disk_percent'] > 90:
            score -= 0.15
        
        # Penalize alerts
        score -= len(health_data.alerts) * 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _send_alerts(self, alerts: List[str]):
        """Envia alertas via webhook"""
        
        if not self.config['webhook_url']:
            self.logger.warning("No webhook URL configured for alerts")
            return
        
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'service': 'crmbet-ultra-robusto',
                'alerts': alerts,
                'severity': 'warning' if len(alerts) <= 2 else 'critical'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['webhook_url'],
                    json=alert_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        self.logger.info("Alerts sent successfully", alerts_count=len(alerts))
                    else:
                        self.logger.error("Failed to send alerts", status=response.status)
        
        except Exception as e:
            self.logger.error("Error sending alerts", error=str(e))
    
    def get_prometheus_metrics(self) -> str:
        """Retorna métricas no formato Prometheus"""
        return generate_latest(self.metrics.registry)

class HealthCheckServer:
    """Servidor HTTP para health checks e métricas"""
    
    def __init__(self, monitoring: UltraRobustMonitoring):
        self.monitoring = monitoring
        self.logger = logger.bind(component="HealthCheckServer")
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 9090):
        """Inicia servidor de health checks"""
        
        from aiohttp import web
        
        app = web.Application()
        
        # Health check endpoints
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/health/detailed', self.detailed_health_check)
        app.router.add_get('/metrics', self.prometheus_metrics)
        app.router.add_get('/performance', self.performance_metrics)
        
        self.logger.info(f"Starting health check server on {host}:{port}")
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        self.logger.info("Health check server started")
    
    async def health_check(self, request):
        """Endpoint básico de health check"""
        
        try:
            health = await self.monitoring.get_comprehensive_health()
            
            status_code = 200 if health.overall_status == 'healthy' else 503
            
            return web.json_response({
                'status': health.overall_status,
                'timestamp': health.timestamp,
                'uptime_seconds': health.uptime_seconds
            }, status=status_code)
        
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'error': str(e)
            }, status=500)
    
    async def detailed_health_check(self, request):
        """Endpoint detalhado de health check"""
        
        try:
            health = await self.monitoring.get_comprehensive_health()
            return web.json_response(asdict(health))
        
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'error': str(e)
            }, status=500)
    
    async def prometheus_metrics(self, request):
        """Endpoint de métricas Prometheus"""
        
        try:
            metrics = self.monitoring.get_prometheus_metrics()
            return web.Response(
                text=metrics,
                content_type=CONTENT_TYPE_LATEST
            )
        
        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def performance_metrics(self, request):
        """Endpoint de métricas de performance"""
        
        try:
            performance = await self.monitoring.get_performance_metrics()
            return web.json_response(asdict(performance))
        
        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)

async def main():
    """Função principal do sistema de monitoramento"""
    
    # Initialize monitoring system
    monitoring = UltraRobustMonitoring()
    
    # Initialize health check server
    health_server = HealthCheckServer(monitoring)
    
    # Start both monitoring and server
    tasks = [
        asyncio.create_task(monitoring.start_monitoring()),
        asyncio.create_task(health_server.start_server())
    ]
    
    try:
        logger.info("🚀 Starting Ultra-Robust Monitoring System")
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Monitoring system stopped by user")
    except Exception as e:
        logger.error("Monitoring system error", error=str(e))
    finally:
        for task in tasks:
            task.cancel()

if __name__ == "__main__":
    asyncio.run(main())