"""
DBSCAN Clusterer - Algoritmo de clusterização baseado em densidade para usuários gaming/apostas
Implementação otimizada para detectar outliers e grupos irregulares de comportamento.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class DBSCANClusterer:
    """
    Algoritmo DBSCAN otimizado para clusterização de usuários gaming/apostas.
    
    Features:
    - Auto-tuning de parâmetros eps e min_samples
    - Detecção inteligente de outliers (usuários anômalos)
    - Identificação de clusters de densidade variável
    - Otimização para campanhas CRM específicas
    """
    
    def __init__(self, 
                 eps_range: Tuple[float, float] = (0.1, 2.0),
                 min_samples_range: Tuple[int, int] = (5, 50),
                 scaler_type: str = 'standard',
                 metric: str = 'euclidean',
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 n_jobs: int = -1):
        """
        Inicializa o DBSCAN Clusterer com configurações otimizadas.
        
        Args:
            eps_range: Range para otimização automática do parâmetro eps
            min_samples_range: Range para otimização do min_samples
            scaler_type: Tipo de scaling ('standard', 'robust', 'minmax')
            metric: Métrica de distância
            algorithm: Algoritmo para nearest neighbors
            leaf_size: Tamanho da folha para BallTree/KDTree
            n_jobs: Número de jobs paralelos
        """
        self.eps_range = eps_range
        self.min_samples_range = min_samples_range
        self.scaler_type = scaler_type
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        
        # Componentes do pipeline
        self.scaler = None
        self.model = None
        self.pca = None
        self.optimal_eps = None
        self.optimal_min_samples = None
        self.cluster_metrics = {}
        self.outlier_analysis = {}
        self.cluster_profiles = {}
        
        # Gaming-specific cluster names para DBSCAN
        self.cluster_names = {
            -1: "Outliers/Anomalous Users",  # DBSCAN usa -1 para outliers
            0: "Core Mainstream Players",
            1: "High Value Density Cluster", 
            2: "Casual Weekend Players",
            3: "Night Gaming Community",
            4: "Sports Betting Focus Group",
            5: "Bonus Optimizers",
            6: "VIP Dense Cluster",
            7: "Mobile Gaming Cluster",
            8: "Risk-Seeking Group",
            9: "Social Gaming Community",
            10: "Retention Challenge Group"
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
        Valida se os dados estão adequados para clustering DBSCAN.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            bool: True se dados são válidos
        """
        if X.empty:
            logger.error("Dataset vazio fornecido")
            return False
            
        if X.shape[0] < 20:
            logger.error("Dataset muito pequeno para DBSCAN (< 20 amostras)")
            return False
            
        if X.isnull().sum().sum() > 0:
            logger.warning("Dados contêm valores nulos - serão tratados")
            
        # Verificar variância das features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        zero_var_cols = X[numeric_cols].var() == 0
        if zero_var_cols.any():
            logger.warning(f"Features com variância zero: {zero_var_cols[zero_var_cols].index.tolist()}")
            
        return True
    
    def _estimate_eps_knn(self, X_scaled: np.ndarray, k: int = 4) -> float:
        """
        Estima o parâmetro eps usando K-distance graph (método dos k vizinhos).
        
        Args:
            X_scaled: Dados normalizados
            k: Número de vizinhos mais próximos
            
        Returns:
            float: Valor estimado para eps
        """
        # Calcular k-distance para cada ponto
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, indices = neighbors_fit.kneighbors(X_scaled)
        
        # Pegar a distância do k-ésimo vizinho mais próximo
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # Encontrar o "cotovelo" na curva k-distance
        # Usar o método da segunda derivada
        if len(k_distances) >= 3:
            first_deriv = np.diff(k_distances)
            second_deriv = np.diff(first_deriv)
            
            # Encontrar ponto de máxima curvatura
            if len(second_deriv) > 0:
                knee_idx = np.argmax(second_deriv)
                eps_estimate = k_distances[knee_idx + 2]  # +2 devido aos diffs
            else:
                eps_estimate = np.percentile(k_distances, 95)  # Fallback
        else:
            eps_estimate = np.percentile(k_distances, 95)
        
        return eps_estimate
    
    def _optimize_parameters(self, X_scaled: np.ndarray) -> Tuple[float, int]:
        """
        Otimiza os parâmetros eps e min_samples do DBSCAN.
        
        Args:
            X_scaled: Dados normalizados
            
        Returns:
            Tuple[float, int]: Valores ótimos de (eps, min_samples)
        """
        # Estimar eps inicial usando k-distance
        estimated_eps = self._estimate_eps_knn(X_scaled)
        logger.info(f"Eps estimado via k-distance: {estimated_eps:.4f}")
        
        # Definir range de eps baseado na estimativa
        eps_min = max(self.eps_range[0], estimated_eps * 0.5)
        eps_max = min(self.eps_range[1], estimated_eps * 2.0)
        eps_values = np.linspace(eps_min, eps_max, 10)
        
        # Range de min_samples baseado no tamanho do dataset
        n_samples = X_scaled.shape[0]
        min_samples_min = max(self.min_samples_range[0], int(np.log(n_samples)))
        min_samples_max = min(self.min_samples_range[1], int(n_samples * 0.05))
        min_samples_values = np.linspace(min_samples_min, min_samples_max, 5, dtype=int)
        
        best_score = -1
        best_eps = estimated_eps
        best_min_samples = min_samples_min
        results = []
        
        logger.info("Otimizando parâmetros DBSCAN via grid search...")
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # Treinar DBSCAN
                    dbscan = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric=self.metric,
                        algorithm=self.algorithm,
                        leaf_size=self.leaf_size,
                        n_jobs=self.n_jobs
                    )
                    
                    labels = dbscan.fit_predict(X_scaled)
                    
                    # Verificar se encontrou clusters válidos
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_outliers = list(labels).count(-1)
                    outlier_ratio = n_outliers / len(labels)
                    
                    # Critérios de qualidade para DBSCAN
                    if n_clusters < 2 or n_clusters > 15 or outlier_ratio > 0.5:
                        continue
                    
                    # Calcular métricas (apenas para pontos não-outliers)
                    non_outlier_mask = labels != -1
                    if np.sum(non_outlier_mask) < 10:  # Muito poucos pontos
                        continue
                    
                    X_non_outliers = X_scaled[non_outlier_mask]
                    labels_non_outliers = labels[non_outlier_mask]
                    
                    if len(set(labels_non_outliers)) > 1:
                        silhouette = silhouette_score(X_non_outliers, labels_non_outliers)
                        davies_bouldin = davies_bouldin_score(X_non_outliers, labels_non_outliers)
                        calinski = calinski_harabasz_score(X_non_outliers, labels_non_outliers)
                        
                        # Score composto (menor é melhor para davies_bouldin)
                        composite_score = silhouette + (1 / (davies_bouldin + 1)) + (calinski / 1000)
                        
                        results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_outliers': n_outliers,
                            'outlier_ratio': outlier_ratio,
                            'silhouette': silhouette,
                            'davies_bouldin': davies_bouldin,
                            'calinski': calinski,
                            'composite_score': composite_score
                        })
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            best_eps = eps
                            best_min_samples = min_samples
                    
                except Exception as e:
                    logger.debug(f"Erro com eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        # Salvar resultados da otimização
        self.cluster_metrics['optimization_results'] = results
        
        if not results:
            # Fallback para valores padrão
            logger.warning("Nenhuma configuração válida encontrada. Usando valores padrão.")
            best_eps = estimated_eps
            best_min_samples = max(5, int(np.log(n_samples)))
        
        logger.info(f"Parâmetros ótimos: eps={best_eps:.4f}, min_samples={best_min_samples}")
        return best_eps, best_min_samples
    
    def fit(self, X: pd.DataFrame) -> 'DBSCANClusterer':
        """
        Treina o modelo DBSCAN com otimização automática.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            self: Instância treinada
        """
        if not self._validate_data(X):
            raise ValueError("Dados inválidos para clustering DBSCAN")
        
        logger.info("Iniciando treinamento do DBSCAN Clusterer")
        
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
        
        # Normalizar dados (crítico para DBSCAN)
        self.scaler = self._initialize_scaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Otimizar parâmetros
        self.optimal_eps, self.optimal_min_samples = self._optimize_parameters(X_scaled)
        
        # Treinar modelo final
        self.model = DBSCAN(
            eps=self.optimal_eps,
            min_samples=self.optimal_min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs
        )
        
        labels = self.model.fit_predict(X_scaled)
        
        # Análise dos resultados
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        
        logger.info(f"Clusters encontrados: {n_clusters}")
        logger.info(f"Outliers detectados: {n_outliers} ({n_outliers/len(labels)*100:.1f}%)")
        
        # Análise de outliers
        self._analyze_outliers(X_processed, labels)
        
        # Gerar perfis dos clusters
        self._generate_cluster_profiles(X_processed, labels)
        
        # Calcular métricas finais
        self._calculate_final_metrics(X_scaled, labels)
        
        # PCA para visualização
        self.pca = PCA(n_components=2, random_state=42)
        self.pca.fit(X_scaled)
        
        logger.info("Treinamento DBSCAN concluído com sucesso")
        return self
    
    def _analyze_outliers(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Analisa características dos outliers detectados."""
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
        
        # Comparar outliers vs não-outliers
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            outlier_stats = {
                'outlier_mean': outliers[col].mean(),
                'outlier_median': outliers[col].median(),
                'outlier_std': outliers[col].std(),
                'non_outlier_mean': non_outliers[col].mean(),
                'non_outlier_median': non_outliers[col].median(),
                'difference': outliers[col].mean() - non_outliers[col].mean()
            }
            analysis['characteristics'][col] = outlier_stats
        
        self.outlier_analysis = analysis
        
        logger.info(f"Análise de outliers: {analysis['count']} usuários anômalos identificados")
    
    def _generate_cluster_profiles(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Gera perfis detalhados de cada cluster incluindo outliers."""
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
                'is_outlier': cluster_id == -1,
                'statistics': {}
            }
            
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
        """Calcula métricas finais do clustering."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_ratio': n_outliers / len(labels),
            'eps': self.optimal_eps,
            'min_samples': self.optimal_min_samples
        }
        
        # Métricas apenas para pontos não-outliers
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
        Prediz clusters para novos dados.
        Nota: DBSCAN não tem método predict nativo, então usa-se aproximação por vizinhos.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            np.ndarray: Array com labels dos clusters
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # DBSCAN não tem método predict, então aproximamos usando nearest neighbors
        logger.warning("DBSCAN não suporta predição direta. Usando aproximação por vizinhos.")
        
        # Processar dados
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_scaled = self.scaler.transform(X_processed)
        
        # Usar pontos de treino para aproximar predições
        if not hasattr(self.model, 'components_'):
            # Se não há core samples, todos são outliers
            return np.full(len(X), -1)
        
        # Encontrar vizinhos mais próximos dos core samples
        core_samples = self.model.components_
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(core_samples)
        
        distances, indices = neighbors.kneighbors(X_scaled)
        
        # Mapear índices para labels
        core_sample_labels = self.model.labels_[self.model.core_sample_indices_]
        predictions = core_sample_labels[indices.flatten()]
        
        # Pontos muito distantes são considerados outliers
        outlier_threshold = self.optimal_eps * 1.5
        outlier_mask = distances.flatten() > outlier_threshold
        predictions[outlier_mask] = -1
        
        return predictions
    
    def get_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna análise detalhada dos outliers identificados.
        
        Args:
            X: DataFrame original usado no treinamento
            
        Returns:
            pd.DataFrame: DataFrame com outliers e suas características
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Reconstruir labels (assumindo mesmo dataset de treino)
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_scaled = self.scaler.transform(X_processed)
        labels = self.model.fit_predict(X_scaled)
        
        # Filtrar outliers
        outlier_mask = labels == -1
        outliers_df = X[outlier_mask].copy()
        outliers_df['cluster'] = -1
        outliers_df['outlier_score'] = self._calculate_outlier_scores(X_scaled[outlier_mask])
        
        return outliers_df
    
    def _calculate_outlier_scores(self, X_outliers: np.ndarray) -> np.ndarray:
        """Calcula scores de anomalia para outliers."""
        if len(X_outliers) == 0:
            return np.array([])
        
        # Usar distância média aos core samples como score
        if hasattr(self.model, 'components_') and len(self.model.components_) > 0:
            neighbors = NearestNeighbors(n_neighbors=min(5, len(self.model.components_)))
            neighbors.fit(self.model.components_)
            distances, _ = neighbors.kneighbors(X_outliers)
            outlier_scores = distances.mean(axis=1)
        else:
            outlier_scores = np.ones(len(X_outliers))
        
        return outlier_scores
    
    def get_cluster_profiles(self) -> Dict[int, Dict]:
        """Retorna perfis detalhados dos clusters."""
        return self.cluster_profiles
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de qualidade do clustering."""
        return self.cluster_metrics
    
    def get_outlier_analysis(self) -> Dict[str, Any]:
        """Retorna análise detalhada dos outliers."""
        return self.outlier_analysis
    
    def plot_clusters(self, X: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plota visualização 2D dos clusters usando PCA, destacando outliers.
        
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
        
        # Obter labels
        labels = self.model.fit_predict(X_scaled)
        
        # Criar plot
        plt.figure(figsize=(14, 10))
        
        unique_labels = set(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Outliers em preto
                class_member_mask = (labels == k)
                xy = X_pca[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=50, 
                           alpha=0.8, label='Outliers/Anomalous')
            else:
                class_member_mask = (labels == k)
                xy = X_pca[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.7,
                           label=self.cluster_names.get(k, f'Cluster {k}'))
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'DBSCAN Clustering - {len(unique_labels)-1} Clusters + Outliers\n'
                 f'eps={self.optimal_eps:.3f}, min_samples={self.optimal_min_samples}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_parameter_optimization(self, save_path: Optional[str] = None) -> None:
        """
        Plota resultados da otimização de parâmetros.
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        if not self.cluster_metrics.get('optimization_results'):
            raise ValueError("Resultados de otimização não disponíveis.")
        
        results = self.cluster_metrics['optimization_results']
        df_results = pd.DataFrame(results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Silhouette Score por parâmetros
        pivot_sil = df_results.pivot_table(values='silhouette', index='eps', columns='min_samples')
        sns.heatmap(pivot_sil, annot=True, fmt='.3f', ax=ax1, cmap='viridis')
        ax1.set_title('Silhouette Score')
        ax1.set_xlabel('min_samples')
        ax1.set_ylabel('eps')
        
        # Davies-Bouldin Score
        pivot_db = df_results.pivot_table(values='davies_bouldin', index='eps', columns='min_samples')
        sns.heatmap(pivot_db, annot=True, fmt='.3f', ax=ax2, cmap='viridis_r')
        ax2.set_title('Davies-Bouldin Score (menor é melhor)')
        ax2.set_xlabel('min_samples')
        ax2.set_ylabel('eps')
        
        # Número de clusters
        pivot_nclusters = df_results.pivot_table(values='n_clusters', index='eps', columns='min_samples')
        sns.heatmap(pivot_nclusters, annot=True, fmt='.0f', ax=ax3, cmap='plasma')
        ax3.set_title('Número de Clusters')
        ax3.set_xlabel('min_samples')
        ax3.set_ylabel('eps')
        
        # Ratio de outliers
        pivot_outliers = df_results.pivot_table(values='outlier_ratio', index='eps', columns='min_samples')
        sns.heatmap(pivot_outliers, annot=True, fmt='.3f', ax=ax4, cmap='Reds')
        ax4.set_title('Proporção de Outliers')
        ax4.set_xlabel('min_samples')
        ax4.set_ylabel('eps')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmaps de otimização salvos em: {save_path}")
        
        plt.show()
    
    def export_cluster_report(self) -> pd.DataFrame:
        """
        Exporta relatório detalhado dos clusters incluindo outliers.
        
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
                'is_outlier': profile['is_outlier']
            }
            
            # Adicionar estatísticas principais
            for feature, stats in profile['statistics'].items():
                row[f'{feature}_mean'] = stats['mean']
                row[f'{feature}_median'] = stats['median']
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)


# Exemplo de uso e teste
if __name__ == "__main__":
    # Dados de exemplo para teste
    np.random.seed(42)
    
    # Simular dados de usuários gaming/apostas com alguns outliers
    n_users = 800
    n_outliers = 50
    
    # Usuários normais
    normal_data = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(2, 1, n_users),
        'total_deposits': np.random.lognormal(4, 1.5, n_users),
        'session_frequency': np.random.poisson(10, n_users),
        'avg_session_duration': np.random.exponential(30, n_users),
        'games_played': np.random.poisson(5, n_users),
        'preferred_hour': np.random.randint(0, 24, n_users),
        'days_since_last_bet': np.random.exponential(3, n_users),
        'win_rate': np.random.beta(2, 3, n_users),
        'cashback_usage': np.random.binomial(1, 0.3, n_users),
        'sports_bet_ratio': np.random.beta(1, 2, n_users)
    })
    
    # Outliers (comportamento anômalo)
    outlier_data = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(5, 2, n_outliers),  # Apostas muito altas
        'total_deposits': np.random.lognormal(7, 2, n_outliers),  # Depósitos anômalos
        'session_frequency': np.random.poisson(50, n_outliers),   # Muitas sessões
        'avg_session_duration': np.random.exponential(200, n_outliers),  # Sessões longas
        'games_played': np.random.poisson(30, n_outliers),       # Muitos jogos
        'preferred_hour': np.random.randint(0, 24, n_outliers),
        'days_since_last_bet': np.random.exponential(1, n_outliers),
        'win_rate': np.random.beta(1, 10, n_outliers),           # Taxa de vitória anômala
        'cashback_usage': np.random.binomial(1, 0.9, n_outliers),
        'sports_bet_ratio': np.random.beta(10, 1, n_outliers)
    })
    
    # Combinar dados
    test_data = pd.concat([normal_data, outlier_data], ignore_index=True)
    
    # Testar DBSCAN Clusterer
    print("🚀 Testando DBSCAN Clusterer...")
    
    clusterer = DBSCANClusterer()
    clusterer.fit(test_data)
    
    metrics = clusterer.get_metrics()
    print(f"✅ Clusters encontrados: {metrics['n_clusters']}")
    print(f"✅ Outliers detectados: {metrics['n_outliers']} ({metrics['outlier_ratio']*100:.1f}%)")
    
    if 'silhouette_score' in metrics:
        print(f"✅ Silhouette Score: {metrics['silhouette_score']:.3f}")
    
    print(f"\n✅ Perfis dos clusters:")
    for cluster_id, profile in clusterer.get_cluster_profiles().items():
        print(f"   {profile['name']}: {profile['size']} usuários ({profile['percentage']:.1f}%)")
    
    # Análise de outliers
    outlier_analysis = clusterer.get_outlier_analysis()
    print(f"\n✅ Análise de outliers: {outlier_analysis['count']} usuários anômalos")
    
    print("\n🎯 DBSCAN Clusterer implementado com sucesso!")