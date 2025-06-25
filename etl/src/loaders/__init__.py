"""
📊 Loaders - Módulos de Carregamento de Dados
Componentes para carregamento otimizado em bancos de dados
"""

from .postgres_loader import PostgresLoader, LoaderConfig, LoadingReport

__all__ = [
    'PostgresLoader',
    'LoaderConfig',
    'LoadingReport'
]