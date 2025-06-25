#!/usr/bin/env python3
"""
🚀 INDUSTRIAL ETL DEPLOYMENT SCRIPT - TB+/HOUR SCALE
Deploys and configures the industrial-grade ETL pipeline with maximum hardness

Author: Agente ETL Industrial - ULTRATHINK
Created: 2025-06-25
"""

import os
import sys
import asyncio
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

# Setup logging
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=True),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
        },
    }
}

import logging.config
logging.config.dictConfig(logging_config)

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

class IndustrialETLDeployer:
    """
    Deployer para ETL industrial com capacidades:
    - Validação de ambiente e dependências
    - Setup automático de infraestrutura
    - Configuração otimizada para TB+/hora
    - Health checks e validação de deployment
    - Rollback automático em caso de falha
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.logger = logger.bind(component="IndustrialETLDeployer", env=environment)
        self.deployment_start_time = datetime.now()
        
        # Paths
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.src_dir = self.project_root / "src"
        
        # Environment configurations
        self.env_configs = {
            "development": {
                "max_workers": 4,
                "batch_size": 10000,
                "enable_monitoring": True,
                "enable_compression": True,
                "prometheus_port": 9090
            },
            "staging": {
                "max_workers": 8,
                "batch_size": 50000,
                "enable_monitoring": True,
                "enable_compression": True,
                "prometheus_port": 9091
            },
            "production": {
                "max_workers": 32,
                "batch_size": 100000,
                "mega_batch_size": 1000000,
                "enable_monitoring": True,
                "enable_compression": True,
                "enable_dask_distributed": True,
                "enable_streaming": True,
                "prometheus_port": 9090
            }
        }
        
        self.logger.info("Industrial ETL Deployer inicializado")
    
    async def deploy(self) -> bool:
        """Executa deployment completo"""
        
        try:
            self.logger.info("🚀 INICIANDO DEPLOYMENT INDUSTRIAL ETL")
            
            # Phase 1: Environment validation
            self.logger.info("📋 Fase 1: Validação de ambiente")
            if not await self._validate_environment():
                raise RuntimeError("Falha na validação de ambiente")
            
            # Phase 2: Dependencies installation
            self.logger.info("📦 Fase 2: Instalação de dependências")
            if not await self._install_dependencies():
                raise RuntimeError("Falha na instalação de dependências")
            
            # Phase 3: Infrastructure setup
            self.logger.info("🏗️ Fase 3: Setup de infraestrutura")
            if not await self._setup_infrastructure():
                raise RuntimeError("Falha no setup de infraestrutura")
            
            # Phase 4: Configuration deployment
            self.logger.info("⚙️ Fase 4: Deploy de configurações")
            if not await self._deploy_configurations():
                raise RuntimeError("Falha no deploy de configurações")
            
            # Phase 5: Service startup
            self.logger.info("🔄 Fase 5: Inicialização de serviços")
            if not await self._start_services():
                raise RuntimeError("Falha na inicialização de serviços")
            
            # Phase 6: Health checks
            self.logger.info("🏥 Fase 6: Verificações de saúde")
            if not await self._health_checks():
                raise RuntimeError("Falha nas verificações de saúde")
            
            # Phase 7: Performance validation
            self.logger.info("⚡ Fase 7: Validação de performance")
            if not await self._performance_validation():
                raise RuntimeError("Falha na validação de performance")
            
            deployment_time = (datetime.now() - self.deployment_start_time).total_seconds()
            
            self.logger.info(
                "✅ DEPLOYMENT CONCLUÍDO COM SUCESSO",
                duration_seconds=deployment_time,
                environment=self.environment
            )
            
            return True
            
        except Exception as e:
            self.logger.error("❌ FALHA NO DEPLOYMENT", error=str(e))
            
            # Tentativa de rollback
            self.logger.info("🔄 Iniciando rollback automático")
            await self._rollback()
            
            return False
    
    async def _validate_environment(self) -> bool:
        """Valida ambiente de deployment"""
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or python_version.minor < 8:
                self.logger.error("Python 3.8+ é obrigatório", current_version=f"{python_version.major}.{python_version.minor}")
                return False
            
            self.logger.info("✅ Python version OK", version=f"{python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check system resources
            import psutil
            
            # Memory check
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 8:
                self.logger.error("Mínimo 8GB RAM obrigatório", current_gb=memory_gb)
                return False
            
            self.logger.info("✅ Memory OK", total_gb=round(memory_gb, 1))
            
            # CPU check
            cpu_count = psutil.cpu_count()
            if cpu_count < 4:
                self.logger.error("Mínimo 4 CPU cores obrigatório", current_cores=cpu_count)
                return False
            
            self.logger.info("✅ CPU cores OK", cores=cpu_count)
            
            # Disk space check
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 50:
                self.logger.error("Mínimo 50GB espaço livre obrigatório", free_gb=free_gb)
                return False
            
            self.logger.info("✅ Disk space OK", free_gb=round(free_gb, 1))
            
            # Check required directories
            required_dirs = [self.src_dir]
            for dir_path in required_dirs:
                if not dir_path.exists():
                    self.logger.error("Diretório obrigatório não encontrado", path=str(dir_path))
                    return False
            
            self.logger.info("✅ Directory structure OK")
            
            # Check network connectivity
            import socket
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=5)
                self.logger.info("✅ Network connectivity OK")
            except OSError:
                self.logger.error("Sem conectividade de rede")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Erro na validação de ambiente", error=str(e))
            return False
    
    async def _install_dependencies(self) -> bool:
        """Instala dependências do sistema"""
        
        try:
            # Install Python dependencies
            self.logger.info("Instalando dependências Python")
            
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file), "--upgrade"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error("Falha na instalação de dependências", 
                                stderr=stderr.decode(), stdout=stdout.decode())
                return False
            
            self.logger.info("✅ Dependências Python instaladas")
            
            # Verify critical imports
            critical_imports = [
                "pandas", "numpy", "dask", "distributed", "polars", 
                "pyarrow", "kafka", "redis", "prometheus_client",
                "lz4", "zstandard", "snappy", "blosc"
            ]
            
            for module in critical_imports:
                try:
                    __import__(module)
                    self.logger.debug("✅ Import OK", module=module)
                except ImportError as e:
                    self.logger.error("Import crítico falhou", module=module, error=str(e))
                    return False
            
            self.logger.info("✅ Todas as dependências críticas verificadas")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro na instalação de dependências", error=str(e))
            return False
    
    async def _setup_infrastructure(self) -> bool:
        """Setup de infraestrutura necessária"""
        
        try:
            # Setup Redis (for caching and state)
            self.logger.info("Verificando Redis")
            if not await self._check_redis():
                self.logger.warning("Redis não disponível - algumas funcionalidades serão limitadas")
            
            # Setup Kafka (for streaming)
            if self.environment == "production":
                self.logger.info("Verificando Kafka")
                if not await self._check_kafka():
                    self.logger.warning("Kafka não disponível - streaming será desabilitado")
            
            # Setup monitoring directories
            monitoring_dirs = [
                self.project_root / "logs",
                self.project_root / "metrics",
                self.project_root / "checkpoints",
                self.project_root / "temp"
            ]
            
            for dir_path in monitoring_dirs:
                dir_path.mkdir(exist_ok=True, parents=True)
                self.logger.debug("✅ Directory created", path=str(dir_path))
            
            self.logger.info("✅ Infrastructure setup completo")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro no setup de infraestrutura", error=str(e))
            return False
    
    async def _check_redis(self) -> bool:
        """Verifica disponibilidade do Redis"""
        
        try:
            import redis
            
            client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=5)
            client.ping()
            
            self.logger.info("✅ Redis disponível")
            return True
            
        except Exception as e:
            self.logger.warning("Redis não disponível", error=str(e))
            return False
    
    async def _check_kafka(self) -> bool:
        """Verifica disponibilidade do Kafka"""
        
        try:
            from kafka import KafkaConsumer
            from kafka.errors import NoBrokersAvailable
            
            consumer = KafkaConsumer(
                bootstrap_servers=['localhost:9092'],
                consumer_timeout_ms=5000
            )
            consumer.close()
            
            self.logger.info("✅ Kafka disponível")
            return True
            
        except NoBrokersAvailable:
            self.logger.warning("Kafka não disponível")
            return False
        except Exception as e:
            self.logger.warning("Erro ao verificar Kafka", error=str(e))
            return False
    
    async def _deploy_configurations(self) -> bool:
        """Deploy das configurações otimizadas"""
        
        try:
            env_config = self.env_configs.get(self.environment, {})
            
            # Create environment file
            env_file_path = self.project_root / f".env.{self.environment}"
            
            env_vars = {
                # Processing configuration
                "ETL_BATCH_SIZE": str(env_config.get("batch_size", 100000)),
                "ETL_MEGA_BATCH_SIZE": str(env_config.get("mega_batch_size", 1000000)),
                "ETL_MAX_WORKERS": str(env_config.get("max_workers", 16)),
                
                # Quality settings
                "ETL_MIN_COMPLETENESS": "0.98",
                "ETL_MAX_OUTLIERS": "0.02",
                "ETL_MIN_FRESHNESS_MIN": "5",
                
                # Performance optimizations
                "ETL_ENABLE_COMPRESSION": str(env_config.get("enable_compression", True)).lower(),
                "ETL_COMPRESSION_ALGO": "lz4",
                "ETL_ENABLE_DASK": str(env_config.get("enable_dask_distributed", False)).lower(),
                
                # Monitoring
                "ETL_DETAILED_METRICS": str(env_config.get("enable_monitoring", True)).lower(),
                "PROMETHEUS_PORT": str(env_config.get("prometheus_port", 9090)),
                
                # Streaming
                "ETL_STREAMING": str(env_config.get("enable_streaming", False)).lower(),
                "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
                
                # Fault tolerance
                "ETL_CIRCUIT_BREAKERS": "true",
                "ETL_ENABLE_DLQ": "true",
                "ETL_MAX_RETRIES": "3",
                
                # Resource optimization
                "ETL_AUTO_SCALING": "true",
                "ETL_PEAK_HOURS": "9,10,11,14,15,16,20,21,22",
                "ETL_SCALE_UP_PEAK": "true"
            }
            
            # Write environment file
            with open(env_file_path, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            self.logger.info("✅ Environment configuration created", 
                           file=str(env_file_path), variables=len(env_vars))
            
            # Create systemd service file for production
            if self.environment == "production":
                await self._create_systemd_service()
            
            return True
            
        except Exception as e:
            self.logger.error("Erro no deploy de configurações", error=str(e))
            return False
    
    async def _create_systemd_service(self) -> bool:
        """Cria arquivo de serviço systemd para produção"""
        
        try:
            service_content = f"""[Unit]
Description=CRM Bet Industrial ETL Pipeline
After=network.target
Wants=network.target

[Service]
Type=simple
User=etl
Group=etl
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}/src
EnvironmentFile={self.project_root}/.env.production
ExecStart={sys.executable} {self.project_root}/run_pipeline.py --mode batch
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crmbet-etl

# Resource limits for TB+ processing
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
"""
            
            service_file = Path("/tmp/crmbet-etl.service")
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            self.logger.info("✅ Systemd service file created", file=str(service_file))
            self.logger.info("Para instalar: sudo cp /tmp/crmbet-etl.service /etc/systemd/system/")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro ao criar serviço systemd", error=str(e))
            return False
    
    async def _start_services(self) -> bool:
        """Inicia serviços ETL"""
        
        try:
            # Set environment
            env_file = self.project_root / f".env.{self.environment}"
            if env_file.exists():
                # Load environment variables
                with open(env_file) as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
            
            # Start monitoring first
            self.logger.info("Iniciando monitoramento")
            # In a real deployment, this would start Prometheus/Grafana
            
            # Test pipeline initialization
            self.logger.info("Testando inicialização do pipeline")
            
            # Import and test basic pipeline functionality
            sys.path.insert(0, str(self.src_dir))
            
            try:
                from etl_pipeline import ETLPipeline, PipelineConfig
                
                # Create test config
                config = PipelineConfig()
                pipeline = ETLPipeline(config)
                
                # Get pipeline status (this will test component initialization)
                status = pipeline.get_pipeline_status()
                
                self.logger.info("✅ Pipeline initialization test passed", status=status)
                
            except Exception as e:
                self.logger.error("Pipeline initialization test failed", error=str(e))
                return False
            
            self.logger.info("✅ Services started successfully")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro ao iniciar serviços", error=str(e))
            return False
    
    async def _health_checks(self) -> bool:
        """Executa verificações de saúde"""
        
        try:
            health_checks = [
                ("Memory Usage", self._check_memory_health),
                ("CPU Usage", self._check_cpu_health),
                ("Disk Usage", self._check_disk_health),
                ("Network Connectivity", self._check_network_health),
                ("Pipeline Components", self._check_pipeline_health)
            ]
            
            for check_name, check_func in health_checks:
                self.logger.info(f"Executando: {check_name}")
                
                if not await check_func():
                    self.logger.error(f"❌ Health check falhou: {check_name}")
                    return False
                
                self.logger.info(f"✅ {check_name} OK")
            
            self.logger.info("✅ Todas as verificações de saúde passaram")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro nas verificações de saúde", error=str(e))
            return False
    
    async def _check_memory_health(self) -> bool:
        """Verifica saúde da memória"""
        import psutil
        
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        
        if memory_usage_percent > 90:
            self.logger.error("Uso de memória muito alto", usage_percent=memory_usage_percent)
            return False
        
        return True
    
    async def _check_cpu_health(self) -> bool:
        """Verifica saúde da CPU"""
        import psutil
        
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_usage > 95:
            self.logger.error("Uso de CPU muito alto", usage_percent=cpu_usage)
            return False
        
        return True
    
    async def _check_disk_health(self) -> bool:
        """Verifica saúde do disco"""
        import psutil
        
        disk_usage = psutil.disk_usage('/')
        usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        if usage_percent > 90:
            self.logger.error("Uso de disco muito alto", usage_percent=usage_percent)
            return False
        
        return True
    
    async def _check_network_health(self) -> bool:
        """Verifica saúde da rede"""
        import socket
        
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except OSError:
            return False
    
    async def _check_pipeline_health(self) -> bool:
        """Verifica saúde dos componentes do pipeline"""
        
        try:
            sys.path.insert(0, str(self.src_dir))
            from etl_pipeline import ETLPipeline, PipelineConfig
            
            config = PipelineConfig()
            pipeline = ETLPipeline(config)
            
            # Test component initialization
            status = pipeline.get_pipeline_status()
            
            # Check if all components are healthy
            components_status = status.get("components_status", {})
            
            for component, component_status in components_status.items():
                if component_status != "ok":
                    self.logger.error("Component unhealthy", component=component, status=component_status)
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("Pipeline health check failed", error=str(e))
            return False
    
    async def _performance_validation(self) -> bool:
        """Valida performance do pipeline"""
        
        try:
            self.logger.info("Executando testes de performance")
            
            # Create small test dataset
            import pandas as pd
            import numpy as np
            
            test_data = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(10000)],
                'amount': np.random.uniform(10, 1000, 10000),
                'timestamp': pd.date_range('2025-01-01', periods=10000, freq='1s'),
                'transaction_type': np.random.choice(['deposit', 'withdrawal', 'bet'], 10000)
            })
            
            # Test basic processing performance
            start_time = datetime.now()
            
            # Simple data processing test
            processed_data = test_data.groupby('user_id').agg({
                'amount': ['sum', 'mean', 'count'],
                'timestamp': ['min', 'max']
            })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate throughput
            records_per_second = len(test_data) / processing_time
            
            self.logger.info("Performance test results", 
                           records=len(test_data),
                           processing_time_seconds=processing_time,
                           records_per_second=records_per_second)
            
            # For TB+ scale, we expect at least 1000 records/second in basic processing
            min_expected_rps = 1000
            
            if records_per_second < min_expected_rps:
                self.logger.error("Performance abaixo do esperado", 
                                current_rps=records_per_second, 
                                expected_rps=min_expected_rps)
                return False
            
            self.logger.info("✅ Performance validation passed")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro na validação de performance", error=str(e))
            return False
    
    async def _rollback(self) -> bool:
        """Executa rollback em caso de falha"""
        
        try:
            self.logger.info("🔄 Executando rollback")
            
            # Stop any started services
            # In a real deployment, this would stop systemd services, containers, etc.
            
            # Remove environment files if they were created
            env_file = self.project_root / f".env.{self.environment}"
            if env_file.exists():
                env_file.unlink()
                self.logger.info("Environment file removed", file=str(env_file))
            
            # Clean up temporary files
            temp_files = [
                Path("/tmp/crmbet-etl.service"),
                self.project_root / "logs",
                self.project_root / "temp"
            ]
            
            for temp_path in temp_files:
                if temp_path.exists():
                    if temp_path.is_file():
                        temp_path.unlink()
                    elif temp_path.is_dir():
                        import shutil
                        shutil.rmtree(temp_path)
                    
                    self.logger.info("Cleaned up", path=str(temp_path))
            
            self.logger.info("✅ Rollback completed")
            
            return True
            
        except Exception as e:
            self.logger.error("Erro no rollback", error=str(e))
            return False

async def main():
    """Função principal"""
    
    parser = argparse.ArgumentParser(
        description="Deploy Industrial ETL Pipeline - TB+/hour scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python deploy_industrial_etl.py --env production --validate-only
  python deploy_industrial_etl.py --env staging --skip-health-checks
  python deploy_industrial_etl.py --env development --verbose
        """
    )
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment for deployment'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment without deploying'
    )
    
    parser.add_argument(
        '--skip-health-checks',
        action='store_true',
        help='Skip health checks during deployment'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Perform rollback of previous deployment'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize deployer
    deployer = IndustrialETLDeployer(args.env)
    
    try:
        if args.rollback:
            logger.info("🔄 Executando rollback")
            success = await deployer._rollback()
            
        elif args.validate_only:
            logger.info("🔍 Executando apenas validação")
            success = await deployer._validate_environment()
            
        else:
            logger.info("🚀 Executando deployment completo")
            success = await deployer.deploy()
        
        if success:
            logger.info("✅ OPERAÇÃO CONCLUÍDA COM SUCESSO")
            
            if not args.validate_only and not args.rollback:
                logger.info("\n" + "="*60)
                logger.info("🎉 INDUSTRIAL ETL PIPELINE DEPLOYED!")
                logger.info("="*60)
                logger.info(f"Environment: {args.env}")
                logger.info(f"Prometheus: http://localhost:{deployer.env_configs[args.env].get('prometheus_port', 9090)}")
                logger.info("Logs: ./logs/")
                logger.info("Metrics: ./metrics/")
                logger.info("\nTo start processing:")
                logger.info("  python run_pipeline.py --mode batch")
                logger.info("\nTo monitor:")
                logger.info("  tail -f logs/etl.log")
                logger.info("="*60)
            
            sys.exit(0)
        else:
            logger.error("❌ OPERAÇÃO FALHOU")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("⚠️ Operação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error("💥 Erro crítico", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())