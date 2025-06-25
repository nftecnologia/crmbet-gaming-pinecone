"""
HDBSCAN Clusterer - Algoritmo hierárquico de clusterização baseado em densidade
Implementação otimizada para usuários gaming/apostas com clusters de densidades variáveis.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
try:
    import hdbscan
except ImportError:
    # Fallback caso HDBSCAN não esteja instalado
    import warnings
    warnings.warn("HDBSCAN não instalado. Instale com: pip install hdbscan")
    hdbscan = None

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class HDBSCANClusterer:
    """
    Algoritmo HDBSCAN otimizado para clusterização hierárquica de usuários gaming/apostas.
    
    Features:
    - Clusterização hierárquica baseada em densidade
    - Detecção robusta de outliers com confidence scores
    - Adaptação automática a clusters de densidades diferentes
    - Interpretação hierárquica dos padrões comportamentais
    """
    
    def __init__(self, 
                 min_cluster_size_range: Tuple[int, int] = (5, 100),
                 min_samples_range: Tuple[int, int] = (1, 20),
                 scaler_type: str = 'standard',
                 metric: str = 'euclidean',
                 cluster_selection_method: str = 'eom',
                 allow_single_cluster: bool = False,
                 prediction_data: bool = True):
        """
        Inicializa o HDBSCAN Clusterer com configurações otimizadas.
        
        Args:
            min_cluster_size_range: Range para otimização do min_cluster_size
            min_samples_range: Range para otimização do min_samples
            scaler_type: Tipo de scaling ('standard', 'robust', 'minmax')
            metric: Métrica de distância
            cluster_selection_method: Método de seleção ('eom' ou 'leaf')
            allow_single_cluster: Permitir cluster único
            prediction_data: Gerar dados para predição
        """
        if hdbscan is None:
            raise ImportError("HDBSCAN não está instalado. Instale com: pip install hdbscan")
        
        self.min_cluster_size_range = min_cluster_size_range
        self.min_samples_range = min_samples_range
        self.scaler_type = scaler_type
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.prediction_data = prediction_data
        
        # Componentes do pipeline
        self.scaler = None
        self.model = None
        self.pca = None
        self.optimal_min_cluster_size = None
        self.optimal_min_samples = None
        self.cluster_metrics = {}
        self.hierarchy_analysis = {}
        self.outlier_analysis = {}
        self.cluster_profiles = {}
        
        # Gaming-specific cluster names para HDBSCAN
        self.cluster_names = {
            -1: "Noise/Outliers",  # HDBSCAN usa -1 para outliers
            0: "Core Gaming Community",
            1: "High Value Segment", 
            2: "Casual Recreational Players",
            3: "Late Night Enthusiasts",
            4: "Sports Betting Specialists",
            5: "Bonus Strategy Players",
            6: "VIP Elite Cluster",
            7: "Mobile First Users",
            8: "Risk Appetite Segment",
            9: "Social Gaming Network",
            10: "Engagement Challenge Group",
            11: "Weekend Warriors",
            12: "Professional Bettors"
        }
    
    def _initialize_scaler(self) -> Any:
        """Inicializa o scaler baseado na configuração."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
    
    def _validate_data(self, X: pd.DataFrame) -> bool:
        """
        Valida se os dados estão adequados para clustering HDBSCAN.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            bool: True se dados são válidos
        """
        if X.empty:
            logger.error("Dataset vazio fornecido")
            return False
            
        if X.shape[0] < 30:
            logger.error("Dataset muito pequeno para HDBSCAN (< 30 amostras)")
            return False
            
        if X.isnull().sum().sum() > 0:
            logger.warning("Dados contêm valores nulos - serão tratados")
            
        # Verificar variância das features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        zero_var_cols = X[numeric_cols].var() == 0
        if zero_var_cols.any():
            logger.warning(f"Features com variância zero: {zero_var_cols[zero_var_cols].index.tolist()}")
            
        return True
    
    def _optimize_parameters(self, X_scaled: np.ndarray) -> Tuple[int, int]:
        """
        Otimiza os parâmetros min_cluster_size e min_samples do HDBSCAN.
        
        Args:
            X_scaled: Dados normalizados
            
        Returns:
            Tuple[int, int]: Valores ótimos de (min_cluster_size, min_samples)
        """
        n_samples = X_scaled.shape[0]
        
        # Ajustar range baseado no tamanho do dataset
        min_cluster_size_min = max(self.min_cluster_size_range[0], int(np.sqrt(n_samples)))
        min_cluster_size_max = min(self.min_cluster_size_range[1], int(n_samples * 0.15))
        min_cluster_sizes = np.linspace(min_cluster_size_min, min_cluster_size_max, 8, dtype=int)
        
        min_samples_min = max(self.min_samples_range[0], 1)
        min_samples_max = min(self.min_samples_range[1], min_cluster_size_min)
        min_samples_values = np.linspace(min_samples_min, min_samples_max, 5, dtype=int)
        
        best_score = -1
        best_min_cluster_size = min_cluster_size_min
        best_min_samples = min_samples_min
        results = []
        
        logger.info(f"Otimizando parâmetros HDBSCAN...")
        logger.info(f"min_cluster_size range: {min_cluster_size_min} - {min_cluster_size_max}")
        logger.info(f"min_samples range: {min_samples_min} - {min_samples_max}")
        
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_values:
                if min_samples > min_cluster_size:
                    continue
                    
                try:
                    # Treinar HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric=self.metric,
                        cluster_selection_method=self.cluster_selection_method,
                        allow_single_cluster=self.allow_single_cluster,
                        prediction_data=self.prediction_data
                    )
                    
                    labels = clusterer.fit_predict(X_scaled)
                    
                    # Verificar se encontrou clusters válidos
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_outliers = list(labels).count(-1)
                    outlier_ratio = n_outliers / len(labels)
                    
                    # Critérios de qualidade
                    if n_clusters < 2 or n_clusters > 20 or outlier_ratio > 0.6:
                        continue
                    
                    # Calcular métricas (apenas para pontos não-outliers)
                    non_outlier_mask = labels != -1
                    if np.sum(non_outlier_mask) < 10:
                        continue
                    
                    X_non_outliers = X_scaled[non_outlier_mask]
                    labels_non_outliers = labels[non_outlier_mask]
                    
                    if len(set(labels_non_outliers)) > 1:
                        silhouette = silhouette_score(X_non_outliers, labels_non_outliers)
                        davies_bouldin = davies_bouldin_score(X_non_outliers, labels_non_outliers)
                        calinski = calinski_harabasz_score(X_non_outliers, labels_non_outliers)
                        
                        # Score específico do HDBSCAN usando cluster persistence
                        persistence_score = np.mean(clusterer.cluster_persistence_) if hasattr(clusterer, 'cluster_persistence_') else 0
                        
                        # Score composto incluindo persistence
                        composite_score = (
                            silhouette * 0.4 + 
                            (1 / (davies_bouldin + 1)) * 0.3 + 
                            (calinski / 1000) * 0.2 +
                            persistence_score * 0.1
                        )
                        
                        results.append({
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_outliers': n_outliers,
                            'outlier_ratio': outlier_ratio,
                            'silhouette': silhouette,
                            'davies_bouldin': davies_bouldin,
                            'calinski': calinski,
                            'persistence_score': persistence_score,
                            'composite_score': composite_score
                        })
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            best_min_cluster_size = min_cluster_size
                            best_min_samples = min_samples
                    
                except Exception as e:
                    logger.debug(f"Erro com min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}")
                    continue
        
        # Salvar resultados da otimização
        self.cluster_metrics['optimization_results'] = results
        
        if not results:
            # Fallback para valores padrão
            logger.warning("Nenhuma configuração válida encontrada. Usando valores padrão.")
            best_min_cluster_size = min_cluster_size_min
            best_min_samples = min_samples_min
        
        logger.info(f"Parâmetros ótimos: min_cluster_size={best_min_cluster_size}, min_samples={best_min_samples}")
        return best_min_cluster_size, best_min_samples
    
    def fit(self, X: pd.DataFrame) -> 'HDBSCANClusterer':
        """
        Treina o modelo HDBSCAN com otimização automática.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            self: Instância treinada
        """
        if not self._validate_data(X):
            raise ValueError("Dados inválidos para clustering HDBSCAN")
        
        logger.info("Iniciando treinamento do HDBSCAN Clusterer")
        
        # Preparar dados
        X_processed = X.copy()
        
        # Tratar valores nulos
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        
        # Remover features com variância zero
        zero_var_cols = X_processed[numeric_cols].var() == 0
        if zero_var_cols.any():
            cols_to_drop = zero_var_cols[zero_var_cols].index.tolist()
            X_processed = X_processed.drop(columns=cols_to_drop)
            logger.info(f"Removidas features com variância zero: {cols_to_drop}")
        
        # Normalizar dados (crítico para HDBSCAN)
        self.scaler = self._initialize_scaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Otimizar parâmetros
        self.optimal_min_cluster_size, self.optimal_min_samples = self._optimize_parameters(X_scaled)
        
        # Treinar modelo final
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.optimal_min_cluster_size,
            min_samples=self.optimal_min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            prediction_data=self.prediction_data
        )
        
        labels = self.model.fit_predict(X_scaled)
        
        # Análise dos resultados
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        
        logger.info(f"Clusters encontrados: {n_clusters}")
        logger.info(f"Outliers detectados: {n_outliers} ({n_outliers/len(labels)*100:.1f}%)")
        
        # Análise hierárquica
        self._analyze_hierarchy()
        
        # Análise de outliers com confidence scores
        self._analyze_outliers_with_scores(X_processed, labels)
        
        # Gerar perfis dos clusters
        self._generate_cluster_profiles(X_processed, labels)
        
        # Calcular métricas finais
        self._calculate_final_metrics(X_scaled, labels)
        
        # PCA para visualização
        self.pca = PCA(n_components=2, random_state=42)
        self.pca.fit(X_scaled)
        
        logger.info("Treinamento HDBSCAN concluído com sucesso")
        return self
    
    def _analyze_hierarchy(self) -> None:
        """Analisa a estrutura hierárquica dos clusters."""
        if not hasattr(self.model, 'condensed_tree_'):
            return
        
        analysis = {
            'has_hierarchy': True,
            'tree_size': len(self.model.condensed_tree_),
            'persistence_available': hasattr(self.model, 'cluster_persistence_')
        }
        
        if analysis['persistence_available']:
            persistence = self.model.cluster_persistence_
            analysis['avg_persistence'] = np.mean(persistence)
            analysis['max_persistence'] = np.max(persistence)
            analysis['min_persistence'] = np.min(persistence)
            analysis['persistence_std'] = np.std(persistence)
        
        # Analisar estabilidade dos clusters
        if hasattr(self.model, 'cluster_persistence_'):
            # Clusters mais estáveis (maior persistence)
            cluster_ids = np.unique(self.model.labels_[self.model.labels_ >= 0])
            if len(cluster_ids) > 0:
                stability_info = {}
                for i, cluster_id in enumerate(cluster_ids):
                    if i < len(persistence):
                        stability_info[cluster_id] = {
                            'persistence': persistence[i],
                            'stability_rank': np.sum(persistence > persistence[i]) + 1
                        }
                analysis['cluster_stability'] = stability_info
        
        self.hierarchy_analysis = analysis
    
    def _analyze_outliers_with_scores(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Analisa outliers incluindo scores de confiança do HDBSCAN."""
        outlier_mask = labels == -1
        
        if not outlier_mask.any():
            self.outlier_analysis = {'count': 0, 'percentage': 0}
            return
        
        outliers = X[outlier_mask]
        non_outliers = X[~outlier_mask]
        
        analysis = {
            'count': outlier_mask.sum(),
            'percentage': outlier_mask.sum() / len(labels) * 100,
            'characteristics': {}
        }
        
        # Scores de confiança dos outliers
        if hasattr(self.model, 'outlier_scores_'):
            outlier_scores = self.model.outlier_scores_[outlier_mask]
            analysis['outlier_scores'] = {
                'mean': np.mean(outlier_scores),
                'median': np.median(outlier_scores),
                'std': np.std(outlier_scores),
                'min': np.min(outlier_scores),
                'max': np.max(outlier_scores)
            }
        
        # Comparar outliers vs não-outliers
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            outlier_stats = {
                'outlier_mean': outliers[col].mean(),
                'outlier_median': outliers[col].median(),
                'outlier_std': outliers[col].std(),
                'non_outlier_mean': non_outliers[col].mean(),
                'non_outlier_median': non_outliers[col].median(),
                'difference': outliers[col].mean() - non_outliers[col].mean(),
                'effect_size': (outliers[col].mean() - non_outliers[col].mean()) / np.sqrt(
                    (outliers[col].var() + non_outliers[col].var()) / 2) if non_outliers[col].var() > 0 else 0
            }
            analysis['characteristics'][col] = outlier_stats
        
        self.outlier_analysis = analysis
        
        logger.info(f"Análise de outliers com scores: {analysis['count']} usuários anômalos")
    
    def _generate_cluster_profiles(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Gera perfis detalhados de cada cluster incluindo métricas hierárquicas."""
        X_with_labels = X.copy()
        X_with_labels['cluster'] = labels
        
        profiles = {}
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            cluster_data = X_with_labels[X_with_labels['cluster'] == cluster_id]
            
            if cluster_data.empty:
                continue
                
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'name': self.cluster_names.get(cluster_id, f"Cluster_{cluster_id}"),
                'is_noise': cluster_id == -1,
                'statistics': {}
            }
            
            # Adicionar informações de estabilidade se disponível
            if (cluster_id != -1 and 
                hasattr(self.model, 'cluster_persistence_') and 
                self.hierarchy_analysis.get('cluster_stability')):
                
                stability_info = self.hierarchy_analysis['cluster_stability'].get(cluster_id)
                if stability_info:
                    profile['persistence'] = stability_info['persistence']
                    profile['stability_rank'] = stability_info['stability_rank']
            
            # Estatísticas por feature
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                profile['statistics'][col] = {
                    'mean': cluster_data[col].mean(),
                    'median': cluster_data[col].median(),
                    'std': cluster_data[col].std(),
                    'min': cluster_data[col].min(),
                    'max': cluster_data[col].max()
                }
            
            profiles[cluster_id] = profile
        
        self.cluster_profiles = profiles
    
    def _calculate_final_metrics(self, X_scaled: np.ndarray, labels: np.ndarray) -> None:
        """Calcula métricas finais do clustering incluindo métricas específicas do HDBSCAN."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_ratio': n_outliers / len(labels),
            'min_cluster_size': self.optimal_min_cluster_size,
            'min_samples': self.optimal_min_samples
        }
        
        # Métricas específicas do HDBSCAN
        if hasattr(self.model, 'cluster_persistence_'):
            metrics['avg_cluster_persistence'] = np.mean(self.model.cluster_persistence_)
            metrics['cluster_persistences'] = self.model.cluster_persistence_.tolist()
        
        if hasattr(self.model, 'probabilities_'):
            membership_strength = self.model.probabilities_[self.model.probabilities_ > 0]
            if len(membership_strength) > 0:
                metrics['avg_membership_strength'] = np.mean(membership_strength)
                metrics['membership_strength_std'] = np.std(membership_strength)
        
        # Métricas tradicionais apenas para pontos não-outliers
        non_outlier_mask = labels != -1
        
        if np.sum(non_outlier_mask) > 10 and n_clusters > 1:
            X_non_outliers = X_scaled[non_outlier_mask]
            labels_non_outliers = labels[non_outlier_mask]
            
            metrics['silhouette_score'] = silhouette_score(X_non_outliers, labels_non_outliers)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_non_outliers, labels_non_outliers)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_non_outliers, labels_non_outliers)
        
        self.cluster_metrics.update(metrics)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz clusters para novos dados usando capacidades nativas do HDBSCAN.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            np.ndarray: Array com labels dos clusters
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        if not hasattr(self.model, 'prediction_data_') or not self.prediction_data:
            logger.warning("Dados de predição não disponíveis. Execute fit() com prediction_data=True.")
            return np.full(len(X), -1)
        
        # Processar dados
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_scaled = self.scaler.transform(X_processed)
        
        # Usar método nativo de predição do HDBSCAN
        test_labels, test_probabilities = hdbscan.approximate_predict(self.model, X_scaled)
        
        return test_labels
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz probabilidades de pertencimento aos clusters.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            np.ndarray: Array com probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        if not hasattr(self.model, 'prediction_data_') or not self.prediction_data:
            logger.warning("Dados de predição não disponíveis.")
            return np.zeros((len(X), 1))
        
        # Processar dados
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_scaled = self.scaler.transform(X_processed)
        
        # Obter probabilidades
        test_labels, test_probabilities = hdbscan.approximate_predict(self.model, X_scaled)
        
        return test_probabilities
    
    def get_cluster_profiles(self) -> Dict[int, Dict]:
        """Retorna perfis detalhados dos clusters."""
        return self.cluster_profiles
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de qualidade do clustering."""
        return self.cluster_metrics
    
    def get_hierarchy_analysis(self) -> Dict[str, Any]:
        """Retorna análise da estrutura hierárquica."""
        return self.hierarchy_analysis
    
    def get_outlier_analysis(self) -> Dict[str, Any]:
        """Retorna análise detalhada dos outliers."""
        return self.outlier_analysis
    
    def plot_clusters(self, X: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plota visualização 2D dos clusters com informações de confiança.
        
        Args:
            X: DataFrame original
            save_path: Caminho para salvar o gráfico
        """
        if self.model is None or self.pca is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Processar dados
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_scaled = self.scaler.transform(X_processed)
        X_pca = self.pca.transform(X_scaled)
        
        # Obter labels e probabilidades
        labels = self.model.labels_
        probabilities = getattr(self.model, 'probabilities_', None)
        
        # Criar plot
        plt.figure(figsize=(14, 10))
        
        unique_labels = set(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Outliers em preto com transparência baseada no outlier score
                class_member_mask = (labels == k)
                xy = X_pca[class_member_mask]
                
                # Usar outlier scores para transparência se disponível
                alpha_values = 0.6
                if hasattr(self.model, 'outlier_scores_'):
                    outlier_scores = self.model.outlier_scores_[class_member_mask]
                    # Normalizar scores para alpha (invertido: maior score = mais transparente)
                    if len(outlier_scores) > 0:
                        alpha_values = 1 - (outlier_scores - outlier_scores.min()) / (outlier_scores.max() - outlier_scores.min() + 1e-8)
                        alpha_values = np.clip(alpha_values, 0.3, 0.8)
                
                plt.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=50, 
                           alpha=alpha_values, label='Noise/Outliers')
            else:
                class_member_mask = (labels == k)
                xy = X_pca[class_member_mask]
                
                # Usar probabilidades para tamanho dos pontos se disponível
                sizes = 50
                alpha_vals = 0.7
                if probabilities is not None:
                    probs = probabilities[class_member_mask]
                    sizes = 30 + probs * 70  # Tamanho entre 30 e 100
                    alpha_vals = 0.4 + probs * 0.5  # Alpha entre 0.4 e 0.9
                
                scatter = plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=sizes, 
                                    alpha=alpha_vals, 
                                    label=self.cluster_names.get(k, f'Cluster {k}'))
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Título com informações hierárquicas
        title = f'HDBSCAN Clustering - {len(unique_labels)-1} Clusters + Noise\n'
        title += f'min_cluster_size={self.optimal_min_cluster_size}, min_samples={self.optimal_min_samples}'
        if self.hierarchy_analysis.get('avg_persistence'):
            title += f', avg_persistence={self.hierarchy_analysis["avg_persistence"]:.3f}'
        
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_cluster_hierarchy(self, save_path: Optional[str] = None) -> None:
        """
        Plota dendrograma da hierarquia de clusters.
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        if not hasattr(self.model, 'condensed_tree_'):
            raise ValueError("Árvore condensada não disponível.")
        
        # Plot da árvore condensada
        plt.figure(figsize=(12, 8))
        self.model.condensed_tree_.plot(select_clusters=True, 
                                       selection_palette=plt.cm.Set3.colors)
        plt.title('HDBSCAN Cluster Hierarchy - Condensed Tree')
        plt.xlabel('Number of Points')
        plt.ylabel('Distance (λ)')
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_hierarchy.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Hierarquia salva em: {save_path.replace('.png', '_hierarchy.png')}")
        
        plt.show()
        
        # Plot de persistência dos clusters se disponível
        if hasattr(self.model, 'cluster_persistence_') and len(self.model.cluster_persistence_) > 0:
            plt.figure(figsize=(10, 6))
            cluster_ids = range(len(self.model.cluster_persistence_))
            plt.bar(cluster_ids, self.model.cluster_persistence_)
            plt.xlabel('Cluster ID')
            plt.ylabel('Persistence')
            plt.title('Cluster Persistence Scores')
            plt.xticks(cluster_ids)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path.replace('.png', '_persistence.png'), dpi=300, bbox_inches='tight')
                logger.info(f"Persistência salva em: {save_path.replace('.png', '_persistence.png')}")
            
            plt.show()
    
    def export_cluster_report(self) -> pd.DataFrame:
        """
        Exporta relatório detalhado dos clusters incluindo métricas hierárquicas.
        
        Returns:
            pd.DataFrame: Relatório dos clusters
        """
        if not self.cluster_profiles:
            raise ValueError("Perfis dos clusters não disponíveis. Execute fit() primeiro.")
        
        report_data = []
        
        for cluster_id, profile in self.cluster_profiles.items():
            row = {
                'cluster_id': cluster_id,
                'cluster_name': profile['name'],
                'size': profile['size'],
                'percentage': profile['percentage'],
                'is_noise': profile['is_noise']
            }
            
            # Adicionar métricas hierárquicas se disponível
            if 'persistence' in profile:
                row['persistence'] = profile['persistence']
                row['stability_rank'] = profile['stability_rank']
            
            # Adicionar estatísticas principais
            for feature, stats in profile['statistics'].items():
                row[f'{feature}_mean'] = stats['mean']
                row[f'{feature}_median'] = stats['median']
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)


# Exemplo de uso e teste
if __name__ == "__main__":
    # Verificar se HDBSCAN está disponível
    if hdbscan is None:
        print("❌ HDBSCAN não instalado. Instale com: pip install hdbscan")
        exit(1)
    
    # Dados de exemplo para teste
    np.random.seed(42)
    
    # Simular dados hierárquicos de usuários gaming/apostas
    n_users = 1000
    n_outliers = 80
    
    # Cluster principal - jogadores regulares
    main_cluster = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(2, 0.8, n_users//2),
        'total_deposits': np.random.lognormal(4, 1.2, n_users//2),
        'session_frequency': np.random.poisson(8, n_users//2),
        'avg_session_duration': np.random.exponential(25, n_users//2),
        'games_played': np.random.poisson(4, n_users//2),
        'preferred_hour': np.random.normal(20, 3, n_users//2),
        'days_since_last_bet': np.random.exponential(2, n_users//2),
        'win_rate': np.random.beta(2, 3, n_users//2),
        'cashback_usage': np.random.binomial(1, 0.4, n_users//2),
        'sports_bet_ratio': np.random.beta(1, 2, n_users//2)
    })
    
    # Sub-cluster - high rollers
    high_roller_cluster = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(4, 1, n_users//3),
        'total_deposits': np.random.lognormal(6, 1.5, n_users//3),
        'session_frequency': np.random.poisson(15, n_users//3),
        'avg_session_duration': np.random.exponential(60, n_users//3),
        'games_played': np.random.poisson(8, n_users//3),
        'preferred_hour': np.random.normal(22, 2, n_users//3),
        'days_since_last_bet': np.random.exponential(1, n_users//3),
        'win_rate': np.random.beta(3, 2, n_users//3),
        'cashback_usage': np.random.binomial(1, 0.7, n_users//3),
        'sports_bet_ratio': np.random.beta(2, 1, n_users//3)
    })
    
    # Casual players
    casual_cluster = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(1, 0.5, n_users//6),
        'total_deposits': np.random.lognormal(2, 0.8, n_users//6),
        'session_frequency': np.random.poisson(3, n_users//6),
        'avg_session_duration': np.random.exponential(15, n_users//6),
        'games_played': np.random.poisson(2, n_users//6),
        'preferred_hour': np.random.uniform(10, 23, n_users//6),
        'days_since_last_bet': np.random.exponential(7, n_users//6),
        'win_rate': np.random.beta(1, 4, n_users//6),
        'cashback_usage': np.random.binomial(1, 0.1, n_users//6),
        'sports_bet_ratio': np.random.beta(1, 3, n_users//6)
    })
    
    # Outliers extremos
    outlier_data = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(6, 3, n_outliers),
        'total_deposits': np.random.lognormal(8, 3, n_outliers),
        'session_frequency': np.random.poisson(100, n_outliers),
        'avg_session_duration': np.random.exponential(300, n_outliers),
        'games_played': np.random.poisson(50, n_outliers),
        'preferred_hour': np.random.randint(0, 24, n_outliers),
        'days_since_last_bet': np.random.exponential(0.1, n_outliers),
        'win_rate': np.random.uniform(0, 1, n_outliers),
        'cashback_usage': np.random.binomial(1, 0.95, n_outliers),
        'sports_bet_ratio': np.random.uniform(0, 1, n_outliers)
    })
    
    # Combinar dados
    test_data = pd.concat([main_cluster, high_roller_cluster, casual_cluster, outlier_data], 
                         ignore_index=True)
    
    # Testar HDBSCAN Clusterer
    print("🚀 Testando HDBSCAN Clusterer...")
    
    clusterer = HDBSCANClusterer(min_cluster_size_range=(10, 150))
    clusterer.fit(test_data)
    
    metrics = clusterer.get_metrics()
    print(f"✅ Clusters encontrados: {metrics['n_clusters']}")
    print(f"✅ Outliers detectados: {metrics['n_outliers']} ({metrics['outlier_ratio']*100:.1f}%)")
    
    if 'silhouette_score' in metrics:
        print(f"✅ Silhouette Score: {metrics['silhouette_score']:.3f}")
    
    # Métricas específicas do HDBSCAN
    if 'avg_cluster_persistence' in metrics:
        print(f"✅ Persistência média dos clusters: {metrics['avg_cluster_persistence']:.3f}")
    
    if 'avg_membership_strength' in metrics:
        print(f"✅ Força média de pertencimento: {metrics['avg_membership_strength']:.3f}")
    
    print(f"\n✅ Perfis dos clusters:")
    for cluster_id, profile in clusterer.get_cluster_profiles().items():
        persistence_info = ""
        if 'persistence' in profile:
            persistence_info = f" (persistence: {profile['persistence']:.3f})"
        print(f"   {profile['name']}: {profile['size']} usuários ({profile['percentage']:.1f}%){persistence_info}")
    
    # Análise hierárquica
    hierarchy = clusterer.get_hierarchy_analysis()
    if hierarchy.get('has_hierarchy'):
        print(f"\n✅ Análise hierárquica: {hierarchy['tree_size']} nós na árvore")
    
    # Teste de predição
    new_users = test_data[:20]
    predictions = clusterer.predict(new_users)
    print(f"\n✅ Predições para 20 novos usuários: {np.unique(predictions, return_counts=True)}")
    
    # Probabilidades se disponível
    try:
        probabilities = clusterer.predict_proba(new_users[:5])
        print(f"✅ Probabilidades de pertencimento (5 primeiros): {probabilities[:5]}")
    except:
        print("⚠️ Probabilidades não disponíveis")
    
    print("\n🎯 HDBSCAN Clusterer implementado com sucesso!")