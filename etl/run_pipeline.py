#!/usr/bin/env python3
"""
🏗️ ETL Pipeline Runner - CRM Bet ML
Script principal para executar o pipeline ETL com todas as configurações

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Adiciona src ao path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Carrega variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

# Importa pipeline
from etl_pipeline import ETLPipeline, PipelineConfig
import structlog

# Configuração de logging
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def load_config_from_env() -> PipelineConfig:
    """Carrega configuração das variáveis de ambiente"""
    
    config = PipelineConfig()
    
    # Database
    if os.getenv('DATABASE_URL'):
        config.db_url = os.getenv('DATABASE_URL')
    
    # Data Lake
    if os.getenv('DATA_LAKE_BUCKET'):
        config.s3_bucket = os.getenv('DATA_LAKE_BUCKET')
    if os.getenv('DATA_LAKE_PREFIX'):
        config.s3_prefix = os.getenv('DATA_LAKE_PREFIX')
    
    # Performance
    if os.getenv('ETL_BATCH_SIZE'):
        config.batch_size = int(os.getenv('ETL_BATCH_SIZE'))
    if os.getenv('ETL_MAX_WORKERS'):
        config.max_workers = int(os.getenv('ETL_MAX_WORKERS'))
    
    # Quality Thresholds
    if os.getenv('MIN_DATA_COMPLETENESS'):
        config.min_data_completeness = float(os.getenv('MIN_DATA_COMPLETENESS'))
    if os.getenv('MAX_OUTLIER_PERCENTAGE'):
        config.max_outlier_percentage = float(os.getenv('MAX_OUTLIER_PERCENTAGE'))
    if os.getenv('MIN_DATA_FRESHNESS_HOURS'):
        config.min_data_freshness_hours = int(os.getenv('MIN_DATA_FRESHNESS_HOURS'))
    
    # Schedule
    if os.getenv('ETL_SCHEDULE'):
        config.run_schedule = os.getenv('ETL_SCHEDULE')
    if os.getenv('ETL_STREAMING'):
        config.enable_streaming = os.getenv('ETL_STREAMING').lower() == 'true'
    
    # Target Features
    if os.getenv('TARGET_FEATURES'):
        config.target_features = os.getenv('TARGET_FEATURES').split(',')
    
    return config

def run_batch_pipeline(config: PipelineConfig, execution_id: Optional[str] = None) -> bool:
    """Executa pipeline em modo batch"""
    
    logger.info("Iniciando pipeline ETL em modo batch")
    
    try:
        # Inicializa pipeline
        pipeline = ETLPipeline(config)
        
        # Executa pipeline completo
        metrics = pipeline.run_full_pipeline(execution_id)
        
        # Log resultados
        logger.info(
            "Pipeline batch executado com sucesso",
            execution_id=metrics.execution_id,
            duration_seconds=metrics.duration_seconds,
            total_records=metrics.total_records_processed,
            success_rate=metrics.success_rate,
            quality_score=metrics.data_quality_score
        )
        
        # Salva relatório
        save_execution_report(metrics)
        
        return True
        
    except Exception as e:
        logger.error("Erro na execução do pipeline batch", error=str(e))
        return False

def run_streaming_pipeline(config: PipelineConfig):
    """Executa pipeline em modo streaming"""
    
    logger.info("Iniciando pipeline ETL em modo streaming")
    
    try:
        pipeline = ETLPipeline(config)
        pipeline.run_streaming_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Pipeline streaming interrompido pelo usuário")
    except Exception as e:
        logger.error("Erro no pipeline streaming", error=str(e))

def run_scheduled_pipeline(config: PipelineConfig):
    """Executa pipeline em modo agendado"""
    
    logger.info("Iniciando pipeline ETL em modo agendado")
    
    try:
        pipeline = ETLPipeline(config)
        pipeline.schedule_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Pipeline agendado interrompido pelo usuário")
    except Exception as e:
        logger.error("Erro no pipeline agendado", error=str(e))

def validate_configuration(config: PipelineConfig) -> bool:
    """Valida configuração do pipeline"""
    
    logger.info("Validando configuração do pipeline")
    
    errors = []
    
    # Validações críticas
    if not config.db_url or config.db_url == "":
        errors.append("DATABASE_URL não configurada")
    
    if not config.s3_bucket or config.s3_bucket == "":
        errors.append("DATA_LAKE_BUCKET não configurado")
    
    # Validações de threshold
    if config.min_data_completeness <= 0 or config.min_data_completeness > 1:
        errors.append("MIN_DATA_COMPLETENESS deve estar entre 0 e 1")
    
    if config.max_outlier_percentage <= 0 or config.max_outlier_percentage > 1:
        errors.append("MAX_OUTLIER_PERCENTAGE deve estar entre 0 e 1")
    
    # Validações de performance
    if config.batch_size <= 0:
        errors.append("ETL_BATCH_SIZE deve ser positivo")
    
    if config.max_workers <= 0:
        errors.append("ETL_MAX_WORKERS deve ser positivo")
    
    if errors:
        logger.error("Erros de configuração encontrados", errors=errors)
        return False
    
    logger.info("Configuração validada com sucesso")
    return True

def save_execution_report(metrics, output_dir: str = "reports"):
    """Salva relatório de execução"""
    
    try:
        # Cria diretório se não existir
        reports_dir = Path(output_dir)
        reports_dir.mkdir(exist_ok=True)
        
        # Nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"etl_execution_report_{timestamp}.json"
        filepath = reports_dir / filename
        
        # Dados do relatório
        report_data = {
            "execution_id": metrics.execution_id,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
            "duration_seconds": metrics.duration_seconds,
            "total_records_processed": metrics.total_records_processed,
            "records_extracted": metrics.records_extracted,
            "records_cleaned": metrics.records_cleaned,
            "records_transformed": metrics.records_transformed,
            "records_loaded": metrics.records_loaded,
            "data_quality_score": metrics.data_quality_score,
            "success_rate": metrics.success_rate,
            "errors": metrics.errors,
            "warnings": metrics.warnings
        }
        
        # Salva arquivo
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Relatório de execução salvo", filepath=str(filepath))
        
    except Exception as e:
        logger.warning("Erro ao salvar relatório de execução", error=str(e))

def check_dependencies():
    """Verifica dependências do sistema"""
    
    logger.info("Verificando dependências do sistema")
    
    required_packages = [
        'pandas', 'numpy', 'sqlalchemy', 'psycopg2', 'boto3', 
        'scikit-learn', 'structlog', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("Dependências faltando", packages=missing_packages)
        logger.info("Execute: pip install -r requirements.txt")
        return False
    
    logger.info("Todas as dependências estão instaladas")
    return True

def main():
    """Função principal"""
    
    parser = argparse.ArgumentParser(
        description='ETL Pipeline CRM Bet ML - Execução com HARDNESS máxima',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_pipeline.py --mode batch --execution-id manual_20250625
  python run_pipeline.py --mode streaming
  python run_pipeline.py --mode schedule
  python run_pipeline.py --validate-only
  python run_pipeline.py --mode batch --dry-run
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['batch', 'streaming', 'schedule'], 
        default='batch',
        help='Modo de execução do pipeline'
    )
    
    parser.add_argument(
        '--execution-id',
        help='ID personalizado para a execução'
    )
    
    parser.add_argument(
        '--config-file',
        help='Arquivo de configuração personalizado'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Apenas valida configuração sem executar pipeline'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Execução de teste sem modificar dados'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Log detalhado'
    )
    
    args = parser.parse_args()
    
    # Verifica dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Carrega configuração
    config = load_config_from_env()
    
    # Valida configuração
    if not validate_configuration(config):
        sys.exit(1)
    
    # Se apenas validação, sai aqui
    if args.validate_only:
        logger.info("Configuração válida - pipeline pronto para execução")
        sys.exit(0)
    
    # Log de início
    logger.info(
        "Iniciando ETL Pipeline",
        mode=args.mode,
        execution_id=args.execution_id,
        dry_run=args.dry_run
    )
    
    try:
        # Executa pipeline baseado no modo
        if args.mode == 'batch':
            success = run_batch_pipeline(config, args.execution_id)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'streaming':
            run_streaming_pipeline(config)
            
        elif args.mode == 'schedule':
            run_scheduled_pipeline(config)
            
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error("Erro crítico na execução", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()