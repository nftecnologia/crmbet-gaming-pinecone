"""
Cluster Model - Modelo principal que unifica todos os algoritmos e feature engines
Implementação completa do pipeline de clusterização para usuários gaming/apostas.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Importar feature engines
import sys
sys.path.append('..')

from features.behavioral_features import BehavioralFeatureEngine
from features.temporal_features import TemporalFeatureEngine
from features.financial_features import FinancialFeatureEngine
from features.communication_features import CommunicationFeatureEngine

# Importar algoritmos de clustering
from clustering.kmeans_clusterer import KMeansClusterer
from clustering.dbscan_clusterer import DBSCANClusterer
from clustering.ensemble_clusterer import EnsembleClusterer

try:
    from clustering.hdbscan_clusterer import HDBSCANClusterer
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class GamingClusterModel:
    """
    Modelo principal de clusterização para usuários gaming/apostas.
    
    Integra todos os feature engines e algoritmos de clustering em um pipeline completo.
    
    Features:
    - Pipeline completo de feature engineering
    - Múltiplos algoritmos de clustering
    - Validação e otimização automática
    - Interpretação business-ready dos clusters
    - Exportação para campanhas CRM
    """
    
    def __init__(self,
                 clustering_algorithm: str = 'ensemble',
                 feature_engines: List[str] = None,
                 model_name: str = 'gaming_cluster_model',
                 cache_features: bool = True,
                 random_state: int = 42):
        """
        Inicializa o modelo de clusterização gaming.
        
        Args:
            clustering_algorithm: Algoritmo principal ('kmeans', 'dbscan', 'hdbscan', 'ensemble')
            feature_engines: Lista de engines a usar ['behavioral', 'temporal', 'financial', 'communication']
            model_name: Nome do modelo para salvamento
            cache_features: Se deve cachear features extraídas
            random_state: Seed para reprodutibilidade
        """
        self.clustering_algorithm = clustering_algorithm
        self.feature_engines = feature_engines or ['behavioral', 'temporal', 'financial', 'communication']
        self.model_name = model_name
        self.cache_features = cache_features
        self.random_state = random_state
        
        # Componentes do modelo
        self.behavioral_engine = BehavioralFeatureEngine()
        self.temporal_engine = TemporalFeatureEngine()
        self.financial_engine = FinancialFeatureEngine()
        self.communication_engine = CommunicationFeatureEngine()
        
        # Clusterer principal
        self.clusterer = None
        self._initialize_clusterer()
        
        # Estado do modelo
        self.is_fitted = False
        self.feature_matrix = None
        self.cluster_labels = None
        self.cluster_profiles = None
        self.feature_importance = None
        self.model_metadata = {}
        
        # Caches
        self._feature_cache = {}
        self._prediction_cache = {}
        
    def _initialize_clusterer(self) -> None:
        """Inicializa o algoritmo de clustering selecionado."""
        logger.info(f"Inicializando algoritmo: {self.clustering_algorithm}")
        
        if self.clustering_algorithm == 'kmeans':
            self.clusterer = KMeansClusterer()
        elif self.clustering_algorithm == 'dbscan':
            self.clusterer = DBSCANClusterer()
        elif self.clustering_algorithm == 'hdbscan':
            if HDBSCAN_AVAILABLE:
                self.clusterer = HDBSCANClusterer()
            else:
                logger.warning("HDBSCAN não disponível. Usando Ensemble.")
                self.clusterer = EnsembleClusterer(use_hdbscan=False)
        elif self.clustering_algorithm == 'ensemble':
            self.clusterer = EnsembleClusterer(use_hdbscan=HDBSCAN_AVAILABLE)
        else:
            raise ValueError(f"Algoritmo não reconhecido: {self.clustering_algorithm}")
    
    def fit(self, 
            df: pd.DataFrame,
            user_id_col: str = 'user_id',
            interaction_data: pd.DataFrame = None,
            validation_split: float = 0.2) -> 'GamingClusterModel':
        """
        Treina o modelo completo de clusterização.
        
        Args:
            df: DataFrame principal com dados dos usuários
            user_id_col: Nome da coluna de user_id
            interaction_data: DataFrame opcional com dados de interações
            validation_split: Proporção para validação
            
        Returns:
            self: Modelo treinado
        """
        logger.info(f"Iniciando treinamento do {self.model_name}")
        logger.info(f"Usuários no dataset: {len(df)}")
        logger.info(f"Feature engines ativos: {self.feature_engines}")
        
        # Validar dados de entrada
        self._validate_input_data(df, user_id_col)
        
        # Extrair features
        self.feature_matrix = self._extract_all_features(df, user_id_col, interaction_data)
        
        # Split para validação se necessário
        if validation_split > 0:
            train_data, val_data = self._train_validation_split(self.feature_matrix, validation_split)
        else:
            train_data = self.feature_matrix
            val_data = None
        
        # Treinar clusterer
        logger.info("Treinando algoritmo de clustering...")
        self.clusterer.fit(train_data)
        
        # Obter labels e profiles
        self.cluster_labels = self.clusterer.predict(self.feature_matrix)
        self.cluster_profiles = self.clusterer.get_cluster_profiles()
        
        # Calcular importância das features
        self._calculate_feature_importance()
        
        # Gerar metadata do modelo
        self._generate_model_metadata(df, val_data)
        
        # Marcar como treinado
        self.is_fitted = True
        
        logger.info(f"Treinamento concluído! Clusters gerados: {len(set(self.cluster_labels))}")
        return self
    
    def _validate_input_data(self, df: pd.DataFrame, user_id_col: str) -> None:
        """Valida dados de entrada."""
        if df.empty:
            raise ValueError("DataFrame vazio fornecido")
        
        if user_id_col not in df.columns:
            logger.warning(f"Coluna {user_id_col} não encontrada. Assumindo dados agregados.")
        
        # Verificar colunas mínimas necessárias
        required_cols = ['bet_amount', 'session_duration']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols and user_id_col in df.columns:
            logger.warning(f"Colunas recomendadas ausentes: {missing_cols}")
    
    def _extract_all_features(self, 
                            df: pd.DataFrame, 
                            user_id_col: str,
                            interaction_data: pd.DataFrame = None) -> pd.DataFrame:
        """Extrai features de todos os engines ativos."""
        logger.info("Extraindo features de todos os engines...")
        
        all_features = pd.DataFrame()
        
        # Determinar índice base
        if user_id_col in df.columns:
            base_index = df[user_id_col].unique()
        else:
            base_index = df.index
        
        all_features = pd.DataFrame(index=base_index)
        
        # Extrair features por engine
        for engine_name in self.feature_engines:
            logger.info(f"Extraindo features: {engine_name}")
            
            try:
                if engine_name == 'behavioral':
                    features = self.behavioral_engine.extract_all_features(df)
                elif engine_name == 'temporal':
                    features = self.temporal_engine.extract_all_features(df)
                elif engine_name == 'financial':
                    features = self.financial_engine.extract_all_features(df)
                elif engine_name == 'communication':
                    features = self.communication_engine.extract_all_features(df, interaction_data=interaction_data)
                else:
                    logger.warning(f"Engine desconhecido: {engine_name}")
                    continue
                
                # Alinhar índices e combinar
                features = features.reindex(all_features.index, fill_value=0)
                all_features = pd.concat([all_features, features], axis=1)
                
                logger.info(f"Features {engine_name}: {len(features.columns)} adicionadas")
                
            except Exception as e:
                logger.error(f"Erro ao extrair features {engine_name}: {e}")
                continue
        
        # Tratar valores ausentes
        all_features = self._handle_missing_values(all_features)
        
        logger.info(f"Feature matrix final: {all_features.shape}")
        return all_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores ausentes na matriz de features."""
        # Estatísticas antes do tratamento
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            logger.info(f"Tratando {missing_before} valores ausentes...")
            
            # Estratégias por tipo de feature
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Para features numéricas, usar mediana
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Para features categóricas, usar moda ou valor padrão
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                if df[col].mode().empty:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            
            # Verificar se ainda há valores ausentes
            missing_after = df.isnull().sum().sum()
            if missing_after > 0:
                logger.warning(f"Ainda há {missing_after} valores ausentes após tratamento")
                df = df.fillna(0)  # Fallback
        
        return df
    
    def _train_validation_split(self, df: pd.DataFrame, validation_split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dados em treino e validação."""
        n_samples = len(df)
        n_val = int(n_samples * validation_split)
        
        # Split aleatório
        np.random.seed(self.random_state)
        val_indices = np.random.choice(df.index, size=n_val, replace=False)
        
        val_data = df.loc[val_indices]
        train_data = df.drop(val_indices)
        
        logger.info(f"Split: {len(train_data)} treino, {len(val_data)} validação")
        return train_data, val_data
    
    def _calculate_feature_importance(self) -> None:
        """Calcula importância das features."""
        if hasattr(self.clusterer, 'get_feature_importance'):
            self.feature_importance = self.clusterer.get_feature_importance()
        else:
            # Fallback: usar variância entre clusters
            if self.cluster_labels is not None:
                importance_scores = {}
                
                for col in self.feature_matrix.columns:
                    feature_values = self.feature_matrix[col]
                    
                    # Calcular variância entre clusters
                    cluster_means = []
                    for cluster_id in set(self.cluster_labels):
                        if cluster_id != -1:  # Excluir outliers
                            cluster_mask = self.cluster_labels == cluster_id
                            cluster_mean = feature_values[cluster_mask].mean()
                            cluster_means.append(cluster_mean)
                    
                    if len(cluster_means) > 1:
                        between_cluster_var = np.var(cluster_means)
                        total_var = feature_values.var()
                        
                        if total_var > 0:
                            importance = between_cluster_var / total_var
                        else:
                            importance = 0
                    else:
                        importance = 0
                    
                    importance_scores[col] = importance
                
                # Ordenar por importância
                self.feature_importance = dict(
                    sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                )
    
    def _generate_model_metadata(self, df: pd.DataFrame, val_data: pd.DataFrame = None) -> None:
        """Gera metadata do modelo."""
        metrics = self.clusterer.get_metrics() if hasattr(self.clusterer, 'get_metrics') else {}
        
        self.model_metadata = {
            'model_name': self.model_name,
            'algorithm': self.clustering_algorithm,
            'feature_engines': self.feature_engines,
            'training_timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(self.feature_matrix.columns),
            'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0),
            'outliers_ratio': list(self.cluster_labels).count(-1) / len(self.cluster_labels),
            'clustering_metrics': metrics,
            'random_state': self.random_state
        }
        
        if val_data is not None:
            self.model_metadata['validation_size'] = len(val_data)
    
    def predict(self, df: pd.DataFrame, 
                user_id_col: str = 'user_id',
                interaction_data: pd.DataFrame = None) -> np.ndarray:
        """
        Prediz clusters para novos dados.
        
        Args:
            df: DataFrame com dados dos usuários
            user_id_col: Nome da coluna de user_id  
            interaction_data: DataFrame opcional com dados de interações
            
        Returns:
            np.ndarray: Array com labels dos clusters
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        logger.info(f"Predizendo clusters para {len(df)} usuários")
        
        # Extrair features usando os mesmos engines
        features = self._extract_all_features(df, user_id_col, interaction_data)
        
        # Alinhar colunas com matriz de treino
        features = self._align_features(features)
        
        # Predizer usando clusterer
        predictions = self.clusterer.predict(features)
        
        logger.info(f"Predições concluídas. Clusters únicos: {np.unique(predictions)}")
        return predictions
    
    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Alinha features com as do modelo treinado."""
        if self.feature_matrix is None:
            raise ValueError("Modelo não possui matriz de features de referência")
        
        # Colunas de referência
        reference_cols = self.feature_matrix.columns
        
        # Adicionar colunas ausentes com valor 0
        missing_cols = set(reference_cols) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
        
        # Remover colunas extras
        extra_cols = set(features.columns) - set(reference_cols)
        if extra_cols:
            features = features.drop(columns=list(extra_cols))
        
        # Reordenar colunas
        features = features[reference_cols]
        
        return features
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos clusters para business.
        
        Returns:
            Dict com resumo executivo dos clusters
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado.")
        
        summary = {
            'model_info': {
                'name': self.model_name,
                'algorithm': self.clustering_algorithm,
                'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0),
                'total_users': len(self.cluster_labels),
                'outliers': list(self.cluster_labels).count(-1)
            },
            'cluster_distribution': {},
            'top_features': {},
            'business_insights': {}
        }
        
        # Distribuição dos clusters
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label != -1:
                cluster_name = self.cluster_profiles.get(label, {}).get('name', f'Cluster_{label}')
                summary['cluster_distribution'][cluster_name] = {
                    'size': int(count),
                    'percentage': float(count / len(self.cluster_labels) * 100)
                }
        
        # Top features
        if self.feature_importance:
            top_features = dict(list(self.feature_importance.items())[:10])
            summary['top_features'] = top_features
        
        # Insights de negócio
        summary['business_insights'] = self._generate_business_insights()
        
        return summary
    
    def _generate_business_insights(self) -> Dict[str, Any]:
        """Gera insights de negócio dos clusters."""
        insights = {}
        
        if self.cluster_profiles:
            # Cluster mais lucrativo (maior ticket médio)
            largest_cluster = None
            highest_value = None
            most_volatile = None
            
            largest_size = 0
            highest_avg_bet = 0
            highest_volatility = 0
            
            for cluster_id, profile in self.cluster_profiles.items():
                if cluster_id == -1:  # Pular outliers
                    continue
                
                size = profile.get('size', 0)
                stats = profile.get('statistics', {})
                
                # Cluster maior
                if size > largest_size:
                    largest_size = size
                    largest_cluster = profile.get('name', f'Cluster_{cluster_id}')
                
                # Cluster de maior valor (aproximação)
                avg_bet_mean = stats.get('spending_avg_bet', {}).get('mean', 0) or stats.get('avg_bet_amount', {}).get('mean', 0)
                if avg_bet_mean > highest_avg_bet:
                    highest_avg_bet = avg_bet_mean
                    highest_value = profile.get('name', f'Cluster_{cluster_id}')
                
                # Cluster mais volátil
                volatility = stats.get('volatility_normalized_volatility', {}).get('mean', 0)
                if volatility > highest_volatility:
                    highest_volatility = volatility
                    most_volatile = profile.get('name', f'Cluster_{cluster_id}')
            
            insights.update({
                'largest_segment': largest_cluster,
                'highest_value_segment': highest_value,
                'most_volatile_segment': most_volatile,
                'total_segments': len([k for k in self.cluster_profiles.keys() if k != -1])
            })
        
        return insights
    
    def export_for_campaigns(self, user_ids: List[Any] = None) -> pd.DataFrame:
        """
        Exporta dados formatados para campanhas CRM.
        
        Args:
            user_ids: Lista de user_ids específicos (opcional)
            
        Returns:
            pd.DataFrame: Dados prontos para campanhas
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado.")
        
        logger.info("Exportando dados para campanhas...")
        
        # Construir DataFrame base
        export_data = []
        
        feature_index = self.feature_matrix.index
        if user_ids:
            # Filtrar usuários específicos
            feature_index = [uid for uid in feature_index if uid in user_ids]
        
        for i, user_id in enumerate(feature_index):
            cluster_label = self.cluster_labels[i]
            
            # Informações do cluster
            cluster_info = self.cluster_profiles.get(cluster_label, {})
            cluster_name = cluster_info.get('name', f'Cluster_{cluster_label}')
            
            # Features importantes para campanhas
            user_features = self.feature_matrix.loc[user_id]
            
            campaign_data = {
                'user_id': user_id,
                'cluster_id': cluster_label,
                'cluster_name': cluster_name,
                'is_outlier': cluster_label == -1,
                'cluster_size': cluster_info.get('size', 0),
                'cluster_percentage': cluster_info.get('percentage', 0)
            }
            
            # Adicionar features mais importantes para campanhas
            important_features = ['spending_avg_bet', 'spending_user_tier', 'engagement_recency_score',
                                'channel_preferred_channel_encoded', 'campaign_overall_response_rate',
                                'timing_preferred_contact_hour', 'financial_roi']
            
            for feature in important_features:
                if feature in user_features.index:
                    campaign_data[feature] = user_features[feature]
            
            export_data.append(campaign_data)
        
        df_export = pd.DataFrame(export_data)
        
        logger.info(f"Dados exportados para {len(df_export)} usuários")
        return df_export
    
    def save_model(self, filepath: str = None) -> str:
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho para salvar (opcional)
            
        Returns:
            str: Caminho onde foi salvo
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.model_name}_{timestamp}.pkl"
        
        # Preparar dados para salvamento
        model_data = {
            'clusterer': self.clusterer,
            'feature_engines': {
                'behavioral': self.behavioral_engine,
                'temporal': self.temporal_engine,
                'financial': self.financial_engine,
                'communication': self.communication_engine
            },
            'feature_matrix': self.feature_matrix,
            'cluster_labels': self.cluster_labels,
            'cluster_profiles': self.cluster_profiles,
            'feature_importance': self.feature_importance,
            'model_metadata': self.model_metadata,
            'config': {
                'clustering_algorithm': self.clustering_algorithm,
                'feature_engines': self.feature_engines,
                'model_name': self.model_name,
                'random_state': self.random_state
            }
        }
        
        # Salvar
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo salvo em: {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GamingClusterModel':
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo
            
        Returns:
            GamingClusterModel: Modelo carregado
        """
        logger.info(f"Carregando modelo de: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Recriar instância
        config = model_data['config']
        model = cls(
            clustering_algorithm=config['clustering_algorithm'],
            feature_engines=config['feature_engines'],
            model_name=config['model_name'],
            random_state=config['random_state']
        )
        
        # Restaurar estado
        model.clusterer = model_data['clusterer']
        model.behavioral_engine = model_data['feature_engines']['behavioral']
        model.temporal_engine = model_data['feature_engines']['temporal'] 
        model.financial_engine = model_data['feature_engines']['financial']
        model.communication_engine = model_data['feature_engines']['communication']
        
        model.feature_matrix = model_data['feature_matrix']
        model.cluster_labels = model_data['cluster_labels']
        model.cluster_profiles = model_data['cluster_profiles']
        model.feature_importance = model_data['feature_importance']
        model.model_metadata = model_data['model_metadata']
        
        model.is_fitted = True
        
        logger.info("Modelo carregado com sucesso")
        return model
    
    def get_model_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo do modelo.
        
        Returns:
            Dict: Relatório detalhado
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado.")
        
        report = {
            'model_metadata': self.model_metadata,
            'cluster_summary': self.get_cluster_summary(),
            'feature_analysis': {
                'total_features': len(self.feature_matrix.columns),
                'top_10_features': dict(list(self.feature_importance.items())[:10]) if self.feature_importance else {},
                'features_by_engine': self._analyze_features_by_engine()
            },
            'data_quality': {
                'missing_values': self.feature_matrix.isnull().sum().sum(),
                'feature_correlations': self._analyze_feature_correlations(),
                'outliers_detected': list(self.cluster_labels).count(-1)
            },
            'clustering_performance': self.clusterer.get_metrics() if hasattr(self.clusterer, 'get_metrics') else {},
            'recommendations': self._generate_model_recommendations()
        }
        
        return report
    
    def _analyze_features_by_engine(self) -> Dict[str, int]:
        """Analisa features por engine."""
        feature_counts = {}
        
        for engine in self.feature_engines:
            engine_features = [col for col in self.feature_matrix.columns if col.startswith(engine)]
            feature_counts[engine] = len(engine_features)
        
        return feature_counts
    
    def _analyze_feature_correlations(self) -> Dict[str, float]:
        """Analisa correlações entre features."""
        numeric_features = self.feature_matrix.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr()
            
            # Encontrar pares com alta correlação
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > 0.8:
                        high_corr_pairs.append(corr_value)
            
            return {
                'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'high_correlation_pairs': len(high_corr_pairs),
                'max_correlation': max(high_corr_pairs) if high_corr_pairs else 0
            }
        else:
            return {'avg_correlation': 0, 'high_correlation_pairs': 0, 'max_correlation': 0}
    
    def _generate_model_recommendations(self) -> List[str]:
        """Gera recomendações para melhoria do modelo."""
        recommendations = []
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        outlier_ratio = list(self.cluster_labels).count(-1) / len(self.cluster_labels)
        
        # Recomendações baseadas no número de clusters
        if n_clusters < 3:
            recommendations.append("Considere adicionar mais features para aumentar a granularidade dos clusters")
        elif n_clusters > 10:
            recommendations.append("Muitos clusters detectados. Considere reduzir features ou ajustar parâmetros")
        
        # Recomendações baseadas em outliers
        if outlier_ratio > 0.3:
            recommendations.append("Alto número de outliers. Verifique qualidade dos dados ou ajuste parâmetros")
        elif outlier_ratio < 0.05:
            recommendations.append("Poucos outliers detectados. Algoritmo pode estar sendo muito permissivo")
        
        # Recomendações baseadas na importância das features
        if self.feature_importance:
            top_importance = list(self.feature_importance.values())[0]
            if top_importance < 0.1:
                recommendations.append("Features têm baixa importância. Considere feature engineering adicional")
        
        if not recommendations:
            recommendations.append("Modelo está bem configurado. Monitore performance em produção")
        
        return recommendations


# Exemplo de uso e teste
if __name__ == "__main__":
    # Dados de exemplo para teste
    np.random.seed(42)
    
    # Simular dados completos de usuários gaming
    n_users = 300
    
    user_data = []
    interaction_data = []
    
    for user_id in range(n_users):
        # Diferentes perfis de usuário
        if user_id < 60:  # High rollers
            base_bet = np.random.lognormal(6, 1)
            session_freq = np.random.poisson(20) + 10
            preferred_hour = np.random.randint(20, 24)
            win_rate = 0.48
        elif user_id < 150:  # Regular players
            base_bet = np.random.lognormal(4, 1)
            session_freq = np.random.poisson(10) + 5
            preferred_hour = np.random.randint(18, 23)
            win_rate = 0.45
        else:  # Casual players
            base_bet = np.random.lognormal(2, 0.8)
            session_freq = np.random.poisson(5) + 1
            preferred_hour = np.random.randint(10, 22)
            win_rate = 0.42
        
        # Dados do usuário
        user_data.append({
            'user_id': user_id,
            'bet_amount': base_bet,
            'avg_bet_amount': base_bet,
            'max_bet_amount': base_bet * np.random.uniform(1.5, 3),
            'session_duration': np.random.exponential(45),
            'session_frequency': session_freq,
            'games_played': np.random.poisson(6),
            'preferred_hour': preferred_hour,
            'win_rate': win_rate,
            'total_deposits': base_bet * session_freq * 0.8,
            'cashback_usage': np.random.binomial(1, 0.4),
            'days_since_last_bet': np.random.exponential(2)
        })
        
        # Dados de interação para communication features
        n_interactions = np.random.poisson(8) + 1
        for _ in range(n_interactions):
            interaction_data.append({
                'user_id': user_id,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 90)),
                'channel': np.random.choice(['email', 'sms', 'push', 'in_app']),
                'type': np.random.choice(['campaign', 'support', 'promotion']),
                'responded': np.random.binomial(1, 0.25),
                'campaign_type': np.random.choice(['welcome', 'retention', 'cashback']),
                'interaction_type': 'campaign_email',
                'content_type': np.random.choice(['promotional', 'educational']),
                'engagement_score': np.random.uniform(0, 1)
            })
    
    user_df = pd.DataFrame(user_data)
    interaction_df = pd.DataFrame(interaction_data)
    
    # Testar modelo completo
    print("🚀 Testando Gaming Cluster Model...")
    
    # Inicializar modelo
    model = GamingClusterModel(
        clustering_algorithm='ensemble',
        feature_engines=['behavioral', 'temporal', 'financial', 'communication'],
        model_name='gaming_crm_model_v1'
    )
    
    # Treinar
    print("Treinando modelo...")
    model.fit(user_df, interaction_data=interaction_df)
    
    # Resumo dos clusters
    summary = model.get_cluster_summary()
    print(f"\n✅ Clusters gerados: {summary['model_info']['n_clusters']}")
    print(f"✅ Total de usuários: {summary['model_info']['total_users']}")
    print(f"✅ Outliers detectados: {summary['model_info']['outliers']}")
    
    # Distribuição dos clusters
    print(f"\n✅ Distribuição dos clusters:")
    for cluster_name, info in summary['cluster_distribution'].items():
        print(f"   {cluster_name}: {info['size']} usuários ({info['percentage']:.1f}%)")
    
    # Features mais importantes
    if summary['top_features']:
        print(f"\n✅ Top 5 features mais importantes:")
        for feature, importance in list(summary['top_features'].items())[:5]:
            print(f"   {feature}: {importance:.3f}")
    
    # Testar predição
    new_users = user_df.sample(20)
    predictions = model.predict(new_users, interaction_data=interaction_df)
    print(f"\n✅ Predições para 20 usuários: {np.unique(predictions, return_counts=True)}")
    
    # Exportar para campanhas
    campaign_data = model.export_for_campaigns()
    print(f"\n✅ Dados exportados para campanhas: {len(campaign_data)} usuários")
    print(f"✅ Colunas disponíveis: {list(campaign_data.columns)}")
    
    # Salvar modelo
    model_path = model.save_model()
    print(f"\n✅ Modelo salvo em: {model_path}")
    
    # Gerar relatório
    report = model.get_model_report()
    print(f"\n✅ Relatório gerado com {len(report)} seções")
    
    print("\n🎯 Gaming Cluster Model implementado com sucesso!")