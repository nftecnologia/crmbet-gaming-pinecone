"""
KMeans Clusterer - Algoritmo de clusterização inteligente para usuários gaming/apostas
Implementação otimizada com validação científica e interpretabilidade para campanhas CRM.

@author: UltraThink Data Science Team
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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

class KMeansClusterer:
    """
    Algoritmo KMeans otimizado para clusterização de usuários gaming/apostas.
    
    Features:
    - Auto-tunning de hiperparâmetros
    - Validação de qualidade dos clusters
    - Interpretação dos perfis comportamentais
    - Otimização para campanhas CRM
    """
    
    def __init__(self, 
                 n_clusters_range: Tuple[int, int] = (3, 12),
                 scaler_type: str = 'standard',
                 random_state: int = 42,
                 max_iter: int = 300,
                 n_init: int = 10):
        """
        Inicializa o KMeans Clusterer com configurações otimizadas.
        
        Args:
            n_clusters_range: Range para otimização automática do número de clusters
            scaler_type: Tipo de scaling ('standard', 'robust', 'minmax')
            random_state: Seed para reprodutibilidade
            max_iter: Máximo de iterações
            n_init: Número de inicializações aleatórias
        """
        self.n_clusters_range = n_clusters_range
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        
        # Componentes do pipeline
        self.scaler = None
        self.model = None
        self.pca = None
        self.optimal_k = None
        self.cluster_metrics = {}
        self.feature_importance = {}
        self.cluster_profiles = {}
        
        # Gaming-specific cluster names
        self.cluster_names = {
            0: "High Roller Crash",
            1: "Night Owl Casino", 
            2: "Weekend Warrior",
            3: "Cashback Lover",
            4: "Sports Bettor",
            5: "Casual Player",
            6: "VIP Whale",
            7: "Morning Bettor",
            8: "Bonus Hunter",
            9: "Social Gamer",
            10: "Risk Seeker",
            11: "Conservative Player"
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
        Valida se os dados estão adequados para clustering.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            bool: True se dados são válidos
        """
        if X.empty:
            logger.error("Dataset vazio fornecido")
            return False
            
        if X.shape[0] < 10:
            logger.error("Dataset muito pequeno (< 10 amostras)")
            return False
            
        if X.isnull().sum().sum() > 0:
            logger.warning("Dados contêm valores nulos - serão tratados")
            
        # Verificar variância das features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        zero_var_cols = X[numeric_cols].var() == 0
        if zero_var_cols.any():
            logger.warning(f"Features com variância zero: {zero_var_cols[zero_var_cols].index.tolist()}")
            
        return True
    
    def _find_optimal_k(self, X_scaled: np.ndarray) -> int:
        """
        Encontra o número ótimo de clusters usando múltiplas métricas.
        
        Args:
            X_scaled: Dados normalizados
            
        Returns:
            int: Número ótimo de clusters
        """
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_scores = []
        
        k_range = range(self.n_clusters_range[0], self.n_clusters_range[1] + 1)
        
        logger.info(f"Otimizando número de clusters no range {self.n_clusters_range}")
        
        for k in k_range:
            # Treinar KMeans
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init
            )
            labels = kmeans.fit_predict(X_scaled)
            
            # Calcular métricas
            inertias.append(kmeans.inertia_)
            
            if k > 1:  # Métricas de validação precisam de pelo menos 2 clusters
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
                calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
            else:
                silhouette_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
                calinski_scores.append(0)
        
        # Método do cotovelo para inércia
        elbow_k = self._find_elbow_point(k_range, inertias)
        
        # Melhores scores de silhueta e Davies-Bouldin
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        best_davies_k = k_range[np.argmin(davies_bouldin_scores)]
        best_calinski_k = k_range[np.argmax(calinski_scores)]
        
        # Combinar resultados com pesos
        k_votes = {k: 0 for k in k_range}
        k_votes[elbow_k] += 2  # Peso maior para método do cotovelo
        k_votes[best_silhouette_k] += 2
        k_votes[best_davies_k] += 1
        k_votes[best_calinski_k] += 1
        
        optimal_k = max(k_votes, key=k_votes.get)
        
        # Salvar métricas para análise
        self.cluster_metrics = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_scores': calinski_scores,
            'elbow_k': elbow_k,
            'best_silhouette_k': best_silhouette_k,
            'best_davies_k': best_davies_k,
            'optimal_k': optimal_k
        }
        
        logger.info(f"Número ótimo de clusters determinado: {optimal_k}")
        return optimal_k
    
    def _find_elbow_point(self, k_range: range, inertias: List[float]) -> int:
        """
        Encontra o ponto do cotovelo na curva de inércia.
        
        Args:
            k_range: Range de valores K
            inertias: Lista de inércias correspondentes
            
        Returns:
            int: K do ponto do cotovelo
        """
        # Método da segunda derivada
        if len(inertias) < 3:
            return k_range[0]
            
        # Calcular derivadas
        first_deriv = np.diff(inertias)
        second_deriv = np.diff(first_deriv)
        
        # Encontrar o ponto de maior curvatura
        if len(second_deriv) > 0:
            elbow_idx = np.argmax(second_deriv) + 2  # +2 devido aos diffs
            return list(k_range)[min(elbow_idx, len(k_range) - 1)]
        
        return k_range[len(k_range) // 2]  # Fallback para meio do range
    
    def fit(self, X: pd.DataFrame) -> 'KMeansClusterer':
        """
        Treina o modelo KMeans com otimização automática.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            self: Instância treinada
        """
        if not self._validate_data(X):
            raise ValueError("Dados inválidos para clustering")
        
        logger.info("Iniciando treinamento do KMeans Clusterer")
        
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
        
        # Normalizar dados
        self.scaler = self._initialize_scaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Encontrar número ótimo de clusters
        self.optimal_k = self._find_optimal_k(X_scaled)
        
        # Treinar modelo final
        self.model = KMeans(
            n_clusters=self.optimal_k,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init
        )
        
        self.model.fit(X_scaled)
        
        # Calcular importância das features
        self._calculate_feature_importance(X_processed, X_scaled)
        
        # Gerar perfis dos clusters
        self._generate_cluster_profiles(X_processed, self.model.labels_)
        
        # PCA para visualização
        self.pca = PCA(n_components=2, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        logger.info(f"Treinamento concluído. Clusters gerados: {self.optimal_k}")
        return self
    
    def _calculate_feature_importance(self, X: pd.DataFrame, X_scaled: np.ndarray) -> None:
        """Calcula importância das features baseado na separação dos clusters."""
        centroids = self.model.cluster_centers_
        
        # Calcular variância entre clusters para cada feature
        between_cluster_var = np.var(centroids, axis=0)
        
        # Calcular variância total de cada feature
        total_var = np.var(X_scaled, axis=0)
        
        # Importância = variância entre clusters / variância total
        importance = between_cluster_var / (total_var + 1e-8)  # Evitar divisão por zero
        
        self.feature_importance = dict(zip(X.columns, importance))
        
        # Ordenar por importância
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    def _generate_cluster_profiles(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Gera perfis detalhados de cada cluster."""
        X_with_labels = X.copy()
        X_with_labels['cluster'] = labels
        
        profiles = {}
        
        for cluster_id in range(self.optimal_k):
            cluster_data = X_with_labels[X_with_labels['cluster'] == cluster_id]
            
            if cluster_data.empty:
                continue
                
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'name': self.cluster_names.get(cluster_id, f"Cluster_{cluster_id}"),
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
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz clusters para novos dados.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            np.ndarray: Array com labels dos clusters
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Processar dados da mesma forma que no treinamento
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        
        # Normalizar
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features ordenada."""
        return self.feature_importance
    
    def get_cluster_profiles(self) -> Dict[int, Dict]:
        """Retorna perfis detalhados dos clusters."""
        return self.cluster_profiles
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de qualidade do clustering."""
        return self.cluster_metrics
    
    def plot_clusters(self, X: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plota visualização 2D dos clusters usando PCA.
        
        Args:
            X: DataFrame original
            save_path: Caminho para salvar o gráfico
        """
        if self.model is None or self.pca is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Transformar dados
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_scaled = self.scaler.transform(X_processed)
        X_pca = self.pca.transform(X_scaled)
        
        # Predizer clusters
        labels = self.predict(X)
        
        # Criar plot
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.optimal_k))
        
        for i in range(self.optimal_k):
            mask = labels == i
            plt.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1], 
                c=[colors[i]], 
                label=self.cluster_names.get(i, f'Cluster {i}'),
                alpha=0.7,
                s=50
            )
        
        # Plot centroids
        centroids_pca = self.pca.transform(self.model.cluster_centers_)
        plt.scatter(
            centroids_pca[:, 0], 
            centroids_pca[:, 1], 
            c='red', 
            marker='x', 
            s=200, 
            linewidths=3,
            label='Centroids'
        )
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'KMeans Clustering - {self.optimal_k} Clusters\nGaming/Apostas User Segmentation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_elbow_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plota curva do cotovelo para análise do número de clusters.
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        if not self.cluster_metrics:
            raise ValueError("Métricas não disponíveis. Execute fit() primeiro.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        k_range = self.cluster_metrics['k_range']
        
        # Inércia (método do cotovelo)
        ax1.plot(k_range, self.cluster_metrics['inertias'], 'bo-')
        ax1.axvline(x=self.cluster_metrics['elbow_k'], color='r', linestyle='--', label=f'Elbow K={self.cluster_metrics["elbow_k"]}')
        ax1.set_xlabel('Número de Clusters (K)')
        ax1.set_ylabel('Inércia')
        ax1.set_title('Método do Cotovelo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette Score
        ax2.plot(k_range, self.cluster_metrics['silhouette_scores'], 'go-')
        ax2.axvline(x=self.cluster_metrics['best_silhouette_k'], color='r', linestyle='--', 
                   label=f'Best K={self.cluster_metrics["best_silhouette_k"]}')
        ax2.set_xlabel('Número de Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Davies-Bouldin Score
        ax3.plot(k_range, self.cluster_metrics['davies_bouldin_scores'], 'ro-')
        ax3.axvline(x=self.cluster_metrics['best_davies_k'], color='g', linestyle='--',
                   label=f'Best K={self.cluster_metrics["best_davies_k"]}')
        ax3.set_xlabel('Número de Clusters (K)')
        ax3.set_ylabel('Davies-Bouldin Score')
        ax3.set_title('Davies-Bouldin Score (menor é melhor)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calinski-Harabasz Score
        ax4.plot(k_range, self.cluster_metrics['calinski_scores'], 'mo-')
        ax4.axvline(x=self.optimal_k, color='black', linestyle='-', linewidth=2,
                   label=f'Optimal K={self.optimal_k}')
        ax4.set_xlabel('Número de Clusters (K)')
        ax4.set_ylabel('Calinski-Harabasz Score')
        ax4.set_title('Calinski-Harabasz Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curva do cotovelo salva em: {save_path}")
        
        plt.show()
    
    def export_cluster_report(self) -> pd.DataFrame:
        """
        Exporta relatório detalhado dos clusters para uso em campanhas.
        
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
                'percentage': profile['percentage']
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
    
    # Simular dados de usuários gaming/apostas
    n_users = 1000
    
    test_data = pd.DataFrame({
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
    
    # Testar KMeans Clusterer
    print("🚀 Testando KMeans Clusterer...")
    
    clusterer = KMeansClusterer(n_clusters_range=(3, 8))
    clusterer.fit(test_data)
    
    print(f"✅ Número ótimo de clusters: {clusterer.optimal_k}")
    print(f"✅ Features mais importantes:")
    
    for feature, importance in list(clusterer.get_feature_importance().items())[:5]:
        print(f"   {feature}: {importance:.3f}")
    
    print(f"\n✅ Perfis dos clusters:")
    for cluster_id, profile in clusterer.get_cluster_profiles().items():
        print(f"   {profile['name']}: {profile['size']} usuários ({profile['percentage']:.1f}%)")
    
    # Predizer novos usuários
    new_users = test_data[:10]
    predictions = clusterer.predict(new_users)
    print(f"\n✅ Predições para novos usuários: {predictions}")
    
    # Exportar relatório
    report = clusterer.export_cluster_report()
    print(f"\n✅ Relatório exportado com {len(report)} clusters")
    
    print("\n🎯 KMeans Clusterer implementado com sucesso!")