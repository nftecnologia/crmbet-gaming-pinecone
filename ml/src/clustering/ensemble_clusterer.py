"""
Ensemble Clusterer - Combinação inteligente de algoritmos de clustering
Implementação que combina KMeans, DBSCAN e HDBSCAN para maximizar precisão e robustez.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import Counter

# Importar nossos clusterers customizados
from .kmeans_clusterer import KMeansClusterer
from .dbscan_clusterer import DBSCANClusterer

try:
    from .hdbscan_clusterer import HDBSCANClusterer
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN não disponível. Ensemble funcionará apenas com KMeans e DBSCAN.")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class EnsembleClusterer:
    """
    Ensemble de algoritmos de clustering que combina múltiplas abordagens para maior robustez.
    
    Features:
    - Combinação inteligente de KMeans, DBSCAN e HDBSCAN
    - Votação ponderada baseada em métricas de qualidade
    - Detecção de consenso entre algoritmos
    - Identificação de usuários estáveis vs instáveis
    - Otimização automática de pesos
    """
    
    def __init__(self, 
                 use_kmeans: bool = True,
                 use_dbscan: bool = True,
                 use_hdbscan: bool = True,
                 consensus_threshold: float = 0.6,
                 weight_strategy: str = 'quality_based',
                 final_clustering_method: str = 'weighted_voting',
                 min_consensus_clusters: int = 2):
        """
        Inicializa o Ensemble Clusterer.
        
        Args:
            use_kmeans: Usar KMeans no ensemble
            use_dbscan: Usar DBSCAN no ensemble
            use_hdbscan: Usar HDBSCAN no ensemble
            consensus_threshold: Threshold para considerar consenso
            weight_strategy: Estratégia de pesos ('equal', 'quality_based', 'adaptive')
            final_clustering_method: Método final ('weighted_voting', 'spectral', 'hybrid')
            min_consensus_clusters: Mínimo de clusters para consenso
        """
        self.use_kmeans = use_kmeans
        self.use_dbscan = use_dbscan
        self.use_hdbscan = use_hdbscan and HDBSCAN_AVAILABLE
        self.consensus_threshold = consensus_threshold
        self.weight_strategy = weight_strategy
        self.final_clustering_method = final_clustering_method
        self.min_consensus_clusters = min_consensus_clusters
        
        # Algoritmos individuais
        self.kmeans_clusterer = None
        self.dbscan_clusterer = None
        self.hdbscan_clusterer = None
        
        # Resultados do ensemble
        self.individual_results = {}
        self.individual_weights = {}
        self.consensus_matrix = None
        self.final_labels = None
        self.ensemble_metrics = {}
        self.cluster_profiles = {}
        self.stability_analysis = {}
        
        # Visualização
        self.pca = None
        
        # Gaming-specific cluster names para ensemble
        self.cluster_names = {
            -1: "Consensus Outliers",
            0: "Mainstream Gaming Community",
            1: "Premium Value Segment", 
            2: "Casual Entertainment Players",
            3: "Night Gaming Enthusiasts",
            4: "Sports Betting Specialists",
            5: "Bonus Optimization Players",
            6: "VIP High-Value Cluster",
            7: "Mobile Gaming Natives",
            8: "Risk-Taking Enthusiasts",
            9: "Social Gaming Community",
            10: "Retention Focus Group",
            11: "Weekend Activity Cluster",
            12: "Professional Gaming Segment"
        }
    
    def _validate_configuration(self) -> bool:
        """Valida se a configuração do ensemble é válida."""
        active_algorithms = sum([self.use_kmeans, self.use_dbscan, self.use_hdbscan])
        
        if active_algorithms < 2:
            logger.error("Ensemble precisa de pelo menos 2 algoritmos ativos")
            return False
        
        if self.use_hdbscan and not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN solicitado mas não disponível. Desabilitando.")
            self.use_hdbscan = False
            active_algorithms -= 1
        
        if active_algorithms < 2:
            logger.error("Após verificações, menos de 2 algoritmos disponíveis")
            return False
        
        return True
    
    def fit(self, X: pd.DataFrame) -> 'EnsembleClusterer':
        """
        Treina o ensemble de clusterers.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            self: Instância treinada
        """
        if not self._validate_configuration():
            raise ValueError("Configuração do ensemble inválida")
        
        logger.info("Iniciando treinamento do Ensemble Clusterer")
        logger.info(f"Algoritmos ativos: KMeans={self.use_kmeans}, DBSCAN={self.use_dbscan}, HDBSCAN={self.use_hdbscan}")
        
        # Treinar algoritmos individuais
        self._train_individual_clusterers(X)
        
        # Calcular pesos baseado na qualidade
        self._calculate_algorithm_weights(X)
        
        # Criar matriz de consenso
        self._create_consensus_matrix(X)
        
        # Gerar clustering final
        self._generate_final_clustering(X)
        
        # Analisar estabilidade
        self._analyze_cluster_stability(X)
        
        # Gerar perfis finais
        self._generate_ensemble_profiles(X)
        
        # Calcular métricas do ensemble
        self._calculate_ensemble_metrics(X)
        
        # PCA para visualização
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_processed = X.select_dtypes(include=[np.number]).fillna(X.select_dtypes(include=[np.number]).median())
        X_scaled = scaler.fit_transform(X_processed)
        self.pca = PCA(n_components=2, random_state=42)
        self.pca.fit(X_scaled)
        
        logger.info("Treinamento do Ensemble concluído com sucesso")
        return self
    
    def _train_individual_clusterers(self, X: pd.DataFrame) -> None:
        """Treina cada algoritmo individual."""
        logger.info("Treinando algoritmos individuais...")
        
        if self.use_kmeans:
            logger.info("Treinando KMeans...")
            self.kmeans_clusterer = KMeansClusterer()
            self.kmeans_clusterer.fit(X)
            self.individual_results['kmeans'] = {
                'labels': self.kmeans_clusterer.model.labels_,
                'metrics': self.kmeans_clusterer.get_metrics(),
                'profiles': self.kmeans_clusterer.get_cluster_profiles()
            }
        
        if self.use_dbscan:
            logger.info("Treinando DBSCAN...")
            self.dbscan_clusterer = DBSCANClusterer()
            self.dbscan_clusterer.fit(X)
            
            # Reproduzir labels para DBSCAN
            X_processed = X.copy()
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
            X_scaled = self.dbscan_clusterer.scaler.transform(X_processed)
            dbscan_labels = self.dbscan_clusterer.model.fit_predict(X_scaled)
            
            self.individual_results['dbscan'] = {
                'labels': dbscan_labels,
                'metrics': self.dbscan_clusterer.get_metrics(),
                'profiles': self.dbscan_clusterer.get_cluster_profiles()
            }
        
        if self.use_hdbscan and HDBSCAN_AVAILABLE:
            logger.info("Treinando HDBSCAN...")
            self.hdbscan_clusterer = HDBSCANClusterer()
            self.hdbscan_clusterer.fit(X)
            self.individual_results['hdbscan'] = {
                'labels': self.hdbscan_clusterer.model.labels_,
                'metrics': self.hdbscan_clusterer.get_metrics(),
                'profiles': self.hdbscan_clusterer.get_cluster_profiles()
            }
    
    def _calculate_algorithm_weights(self, X: pd.DataFrame) -> None:
        """Calcula pesos para cada algoritmo baseado na qualidade."""
        logger.info(f"Calculando pesos usando estratégia: {self.weight_strategy}")
        
        if self.weight_strategy == 'equal':
            # Pesos iguais
            n_algorithms = len(self.individual_results)
            weight = 1.0 / n_algorithms
            for algo in self.individual_results.keys():
                self.individual_weights[algo] = weight
        
        elif self.weight_strategy == 'quality_based':
            # Pesos baseados em métricas de qualidade
            quality_scores = {}
            
            for algo, results in self.individual_results.items():
                metrics = results['metrics']
                
                # Calcular score composto
                score = 0
                weight_count = 0
                
                # Silhouette score (maior é melhor)
                if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
                    score += metrics['silhouette_score'] * 0.4
                    weight_count += 0.4
                
                # Davies-Bouldin score (menor é melhor)
                if 'davies_bouldin_score' in metrics and metrics['davies_bouldin_score'] is not None:
                    score += (1 / (metrics['davies_bouldin_score'] + 1)) * 0.3
                    weight_count += 0.3
                
                # Calinski-Harabasz score (maior é melhor)
                if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
                    score += (metrics['calinski_harabasz_score'] / 1000) * 0.2
                    weight_count += 0.2
                
                # Penalizar outlier ratio muito alto ou muito baixo
                if 'outlier_ratio' in metrics:
                    outlier_penalty = abs(metrics['outlier_ratio'] - 0.1)  # 10% é ideal
                    score += (1 - outlier_penalty) * 0.1
                    weight_count += 0.1
                
                # Normalizar pelo peso total usado
                if weight_count > 0:
                    quality_scores[algo] = score / weight_count
                else:
                    quality_scores[algo] = 0.5  # Score neutro
            
            # Normalizar pesos
            total_quality = sum(quality_scores.values())
            if total_quality > 0:
                for algo in quality_scores:
                    self.individual_weights[algo] = quality_scores[algo] / total_quality
            else:
                # Fallback para pesos iguais
                n_algorithms = len(self.individual_results)
                for algo in self.individual_results.keys():
                    self.individual_weights[algo] = 1.0 / n_algorithms
        
        elif self.weight_strategy == 'adaptive':
            # Pesos adaptativos baseados em consenso
            # Implementação mais complexa que considera concordância entre algoritmos
            self._calculate_adaptive_weights(X)
        
        logger.info(f"Pesos calculados: {self.individual_weights}")
    
    def _calculate_adaptive_weights(self, X: pd.DataFrame) -> None:
        """Calcula pesos adaptativos baseados em consenso e qualidade."""
        algorithms = list(self.individual_results.keys())
        n_algorithms = len(algorithms)
        consensus_scores = {}
        
        # Calcular consenso par-a-par
        for i, algo1 in enumerate(algorithms):
            consensus_scores[algo1] = 0
            labels1 = self.individual_results[algo1]['labels']
            
            for j, algo2 in enumerate(algorithms):
                if i != j:
                    labels2 = self.individual_results[algo2]['labels']
                    # Usar ARI (Adjusted Rand Index) para medir consenso
                    ari = adjusted_rand_score(labels1, labels2)
                    consensus_scores[algo1] += max(0, ari)  # Apenas consenso positivo
        
        # Combinar com scores de qualidade
        quality_weights = {}
        self.weight_strategy = 'quality_based'  # Temporário
        self._calculate_algorithm_weights(X)
        quality_weights = self.individual_weights.copy()
        
        # Pesos finais: 70% qualidade + 30% consenso
        total_consensus = sum(consensus_scores.values())
        if total_consensus > 0:
            for algo in algorithms:
                consensus_weight = consensus_scores[algo] / total_consensus
                final_weight = quality_weights[algo] * 0.7 + consensus_weight * 0.3
                self.individual_weights[algo] = final_weight
        else:
            # Fallback para qualidade apenas
            self.individual_weights = quality_weights
        
        # Normalizar
        total_weight = sum(self.individual_weights.values())
        if total_weight > 0:
            for algo in self.individual_weights:
                self.individual_weights[algo] /= total_weight
    
    def _create_consensus_matrix(self, X: pd.DataFrame) -> None:
        """Cria matriz de consenso entre algoritmos."""
        n_samples = len(X)
        algorithms = list(self.individual_results.keys())
        
        # Matriz de consenso: consenso[i,j] = grau de concordância para amostra i vs j
        self.consensus_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    self.consensus_matrix[i, j] = 1.0
                else:
                    consensus_count = 0
                    total_algorithms = 0
                    
                    for algo in algorithms:
                        labels = self.individual_results[algo]['labels']
                        weight = self.individual_weights[algo]
                        
                        # Verificar se amostras i e j estão no mesmo cluster
                        if labels[i] == labels[j] and labels[i] != -1:
                            consensus_count += weight
                        total_algorithms += weight
                    
                    if total_algorithms > 0:
                        self.consensus_matrix[i, j] = consensus_count / total_algorithms
    
    def _generate_final_clustering(self, X: pd.DataFrame) -> None:
        """Gera clustering final baseado no método escolhido."""
        logger.info(f"Gerando clustering final usando método: {self.final_clustering_method}")
        
        if self.final_clustering_method == 'weighted_voting':
            self._weighted_voting_clustering(X)
        elif self.final_clustering_method == 'spectral':
            self._spectral_ensemble_clustering(X)
        elif self.final_clustering_method == 'hybrid':
            self._hybrid_ensemble_clustering(X)
        else:
            raise ValueError(f"Método não reconhecido: {self.final_clustering_method}")
    
    def _weighted_voting_clustering(self, X: pd.DataFrame) -> None:
        """Clustering por votação ponderada."""
        n_samples = len(X)
        
        # Mapear labels para IDs consistentes
        label_mappings = {}
        max_cluster_id = -1
        
        for algo, results in self.individual_results.items():
            labels = results['labels']
            unique_labels = sorted(set(labels))
            mapping = {}
            
            for label in unique_labels:
                if label == -1:  # Outliers mantêm -1
                    mapping[label] = -1
                else:
                    max_cluster_id += 1
                    mapping[label] = max_cluster_id
            
            label_mappings[algo] = mapping
        
        # Votação ponderada para cada amostra
        final_labels = []
        
        for i in range(n_samples):
            votes = {}  # cluster_id -> peso total
            
            for algo, results in self.individual_results.items():
                original_label = results['labels'][i]
                mapped_label = label_mappings[algo][original_label]
                weight = self.individual_weights[algo]
                
                if mapped_label in votes:
                    votes[mapped_label] += weight
                else:
                    votes[mapped_label] = weight
            
            # Escolher cluster com maior peso
            if votes:
                best_cluster = max(votes.keys(), key=lambda k: votes[k])
                # Só aceitar se tiver consenso mínimo
                if votes[best_cluster] >= self.consensus_threshold:
                    final_labels.append(best_cluster)
                else:
                    final_labels.append(-1)  # Sem consenso = outlier
            else:
                final_labels.append(-1)
        
        # Remapear para IDs sequenciais
        unique_final = sorted(set(final_labels))
        if -1 in unique_final:
            unique_final.remove(-1)
        
        remap = {-1: -1}
        for i, label in enumerate(unique_final):
            remap[label] = i
        
        self.final_labels = np.array([remap[label] for label in final_labels])
    
    def _spectral_ensemble_clustering(self, X: pd.DataFrame) -> None:
        """Clustering usando Spectral Clustering na matriz de consenso."""
        # Usar matriz de consenso como matriz de afinidade
        n_clusters = self._estimate_optimal_clusters()
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        self.final_labels = spectral.fit_predict(self.consensus_matrix)
    
    def _hybrid_ensemble_clustering(self, X: pd.DataFrame) -> None:
        """Método híbrido que combina votação ponderada e spectral."""
        # Primeiro, usar votação ponderada
        self._weighted_voting_clustering(X)
        voting_labels = self.final_labels.copy()
        
        # Depois, refinar com spectral clustering
        n_clusters = len(set(voting_labels)) - (1 if -1 in voting_labels else 0)
        if n_clusters >= 2:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            spectral_labels = spectral.fit_predict(self.consensus_matrix)
            
            # Combinar resultados: usar spectral apenas onde votação foi incerta
            uncertain_mask = voting_labels == -1
            self.final_labels[uncertain_mask] = spectral_labels[uncertain_mask]
    
    def _estimate_optimal_clusters(self) -> int:
        """Estima número ótimo de clusters baseado nos algoritmos individuais."""
        cluster_counts = []
        
        for algo, results in self.individual_results.items():
            labels = results['labels']
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_counts.append(n_clusters)
        
        # Usar mediana como estimativa robusta
        optimal_k = int(np.median(cluster_counts))
        return max(optimal_k, self.min_consensus_clusters)
    
    def _analyze_cluster_stability(self, X: pd.DataFrame) -> None:
        """Analisa estabilidade dos clusters finais."""
        if self.final_labels is None:
            return
        
        n_samples = len(X)
        stability_scores = np.zeros(n_samples)
        
        # Para cada amostra, calcular quantos algoritmos concordam com o label final
        for i in range(n_samples):
            final_label = self.final_labels[i]
            agreement_count = 0
            total_weight = 0
            
            for algo, results in self.individual_results.items():
                weight = self.individual_weights[algo]
                algo_label = results['labels'][i]
                
                # Verificar se há concordância (mesmo cluster ou ambos outliers)
                if ((final_label == -1 and algo_label == -1) or 
                    (final_label != -1 and algo_label != -1)):
                    agreement_count += weight
                total_weight += weight
            
            if total_weight > 0:
                stability_scores[i] = agreement_count / total_weight
        
        # Análises por cluster
        cluster_stability = {}
        unique_labels = set(self.final_labels)
        
        for label in unique_labels:
            mask = self.final_labels == label
            cluster_stabilities = stability_scores[mask]
            
            cluster_stability[label] = {
                'mean_stability': np.mean(cluster_stabilities),
                'median_stability': np.median(cluster_stabilities),
                'std_stability': np.std(cluster_stabilities),
                'min_stability': np.min(cluster_stabilities),
                'stable_samples': np.sum(cluster_stabilities > 0.8),
                'unstable_samples': np.sum(cluster_stabilities < 0.5)
            }
        
        self.stability_analysis = {
            'sample_stability': stability_scores,
            'cluster_stability': cluster_stability,
            'overall_stability': np.mean(stability_scores)
        }
        
        logger.info(f"Estabilidade geral do ensemble: {self.stability_analysis['overall_stability']:.3f}")
    
    def _generate_ensemble_profiles(self, X: pd.DataFrame) -> None:
        """Gera perfis dos clusters finais."""
        if self.final_labels is None:
            return
        
        X_with_labels = X.copy()
        X_with_labels['cluster'] = self.final_labels
        
        profiles = {}
        unique_labels = set(self.final_labels)
        
        for cluster_id in unique_labels:
            cluster_data = X_with_labels[X_with_labels['cluster'] == cluster_id]
            
            if cluster_data.empty:
                continue
                
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'name': self.cluster_names.get(cluster_id, f"Ensemble_Cluster_{cluster_id}"),
                'is_outlier': cluster_id == -1,
                'statistics': {},
                'stability': self.stability_analysis['cluster_stability'].get(cluster_id, {})
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
    
    def _calculate_ensemble_metrics(self, X: pd.DataFrame) -> None:
        """Calcula métricas do ensemble."""
        if self.final_labels is None:
            return
        
        n_clusters = len(set(self.final_labels)) - (1 if -1 in self.final_labels else 0)
        n_outliers = list(self.final_labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_ratio': n_outliers / len(self.final_labels),
            'algorithm_weights': self.individual_weights.copy(),
            'consensus_threshold': self.consensus_threshold,
            'overall_stability': self.stability_analysis.get('overall_stability', 0)
        }
        
        # Métricas de qualidade (apenas pontos não-outliers)
        non_outlier_mask = self.final_labels != -1
        
        if np.sum(non_outlier_mask) > 10 and n_clusters > 1:
            # Precisamos processar X da mesma forma
            X_processed = X.copy()
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            X_non_outliers = X_scaled[non_outlier_mask]
            labels_non_outliers = self.final_labels[non_outlier_mask]
            
            if len(set(labels_non_outliers)) > 1:
                metrics['silhouette_score'] = silhouette_score(X_non_outliers, labels_non_outliers)
                
                from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_non_outliers, labels_non_outliers)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_non_outliers, labels_non_outliers)
        
        # Consenso entre algoritmos individuais
        consensus_metrics = {}
        algorithms = list(self.individual_results.keys())
        
        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i < j:
                    labels1 = self.individual_results[algo1]['labels']
                    labels2 = self.individual_results[algo2]['labels']
                    
                    ari = adjusted_rand_score(labels1, labels2)
                    ami = adjusted_mutual_info_score(labels1, labels2)
                    
                    consensus_metrics[f'{algo1}_vs_{algo2}'] = {
                        'adjusted_rand_score': ari,
                        'adjusted_mutual_info': ami
                    }
        
        metrics['inter_algorithm_consensus'] = consensus_metrics
        
        self.ensemble_metrics = metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz clusters para novos dados usando ensemble.
        
        Args:
            X: DataFrame com features dos usuários
            
        Returns:
            np.ndarray: Array com labels dos clusters
        """
        if self.final_labels is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        predictions = []
        
        # Obter predições de cada algoritmo
        individual_predictions = {}
        
        if self.use_kmeans and self.kmeans_clusterer:
            individual_predictions['kmeans'] = self.kmeans_clusterer.predict(X)
        
        if self.use_dbscan and self.dbscan_clusterer:
            individual_predictions['dbscan'] = self.dbscan_clusterer.predict(X)
        
        if self.use_hdbscan and self.hdbscan_clusterer:
            individual_predictions['hdbscan'] = self.hdbscan_clusterer.predict(X)
        
        # Votação ponderada para cada amostra
        n_samples = len(X)
        ensemble_predictions = []
        
        for i in range(n_samples):
            votes = {}
            
            for algo, preds in individual_predictions.items():
                if i < len(preds):
                    pred = preds[i]
                    weight = self.individual_weights.get(algo, 0)
                    
                    if pred in votes:
                        votes[pred] += weight
                    else:
                        votes[pred] = weight
            
            # Escolher predição com maior peso
            if votes:
                best_pred = max(votes.keys(), key=lambda k: votes[k])
                if votes[best_pred] >= self.consensus_threshold:
                    ensemble_predictions.append(best_pred)
                else:
                    ensemble_predictions.append(-1)  # Sem consenso
            else:
                ensemble_predictions.append(-1)
        
        return np.array(ensemble_predictions)
    
    def get_cluster_profiles(self) -> Dict[int, Dict]:
        """Retorna perfis detalhados dos clusters do ensemble."""
        return self.cluster_profiles
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do ensemble."""
        return self.ensemble_metrics
    
    def get_individual_results(self) -> Dict[str, Dict]:
        """Retorna resultados dos algoritmos individuais."""
        return self.individual_results
    
    def get_stability_analysis(self) -> Dict[str, Any]:
        """Retorna análise de estabilidade."""
        return self.stability_analysis
    
    def plot_ensemble_comparison(self, X: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plota comparação entre algoritmos individuais e ensemble.
        
        Args:
            X: DataFrame original
            save_path: Caminho para salvar o gráfico
        """
        if self.pca is None or self.final_labels is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Processar dados para PCA
        X_processed = X.select_dtypes(include=[np.number]).fillna(X.select_dtypes(include=[np.number]).median())
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        X_pca = self.pca.transform(X_scaled)
        
        # Configurar subplot
        n_algorithms = len(self.individual_results) + 1  # +1 para ensemble
        n_cols = min(3, n_algorithms)
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_algorithms == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_algorithms == 1 else axes
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot algoritmos individuais
        for algo, results in self.individual_results.items():
            ax = axes[plot_idx]
            labels = results['labels']
            
            unique_labels = set(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    mask = labels == k
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', 
                              s=30, alpha=0.6, label='Outliers')
                else:
                    mask = labels == k
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], s=30, 
                              alpha=0.7, label=f'C{k}')
            
            ax.set_title(f'{algo.upper()}\n(peso: {self.individual_weights.get(algo, 0):.2f})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot ensemble
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            labels = self.final_labels
            
            unique_labels = set(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    mask = labels == k
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', 
                              s=30, alpha=0.6, label='Outliers')
                else:
                    mask = labels == k
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], s=30, 
                              alpha=0.7, label=f'C{k}')
            
            ax.set_title(f'ENSEMBLE\n(estabilidade: {self.stability_analysis.get("overall_stability", 0):.2f})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
        
        # Remover subplots vazios
        for idx in range(plot_idx + 1, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparação salva em: {save_path}")
        
        plt.show()
    
    def plot_stability_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plota análise de estabilidade dos clusters.
        
        Args:
            save_path: Caminho para salvar o gráfico
        """
        if not self.stability_analysis:
            raise ValueError("Análise de estabilidade não disponível.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histograma de estabilidade das amostras
        stability_scores = self.stability_analysis['sample_stability']
        ax1.hist(stability_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(stability_scores), color='red', linestyle='--', 
                   label=f'Média: {np.mean(stability_scores):.3f}')
        ax1.set_xlabel('Score de Estabilidade')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Distribuição de Estabilidade das Amostras')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Estabilidade por cluster
        cluster_stability = self.stability_analysis['cluster_stability']
        cluster_ids = list(cluster_stability.keys())
        mean_stabilities = [cluster_stability[cid]['mean_stability'] for cid in cluster_ids]
        
        ax2.bar(range(len(cluster_ids)), mean_stabilities, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Estabilidade Média')
        ax2.set_title('Estabilidade por Cluster')
        ax2.set_xticks(range(len(cluster_ids)))
        ax2.set_xticklabels([str(cid) for cid in cluster_ids])
        ax2.grid(True, alpha=0.3)
        
        # Pesos dos algoritmos
        algorithms = list(self.individual_weights.keys())
        weights = list(self.individual_weights.values())
        
        ax3.pie(weights, labels=algorithms, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Distribuição de Pesos dos Algoritmos')
        
        # Consenso entre algoritmos
        if 'inter_algorithm_consensus' in self.ensemble_metrics:
            consensus_data = self.ensemble_metrics['inter_algorithm_consensus']
            comparisons = list(consensus_data.keys())
            ari_scores = [consensus_data[comp]['adjusted_rand_score'] for comp in comparisons]
            
            ax4.bar(range(len(comparisons)), ari_scores, color='orange', alpha=0.7)
            ax4.set_xlabel('Comparação de Algoritmos')
            ax4.set_ylabel('Adjusted Rand Index')
            ax4.set_title('Consenso entre Algoritmos')
            ax4.set_xticks(range(len(comparisons)))
            ax4.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Dados de consenso\nnão disponíveis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Consenso entre Algoritmos')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Análise de estabilidade salva em: {save_path}")
        
        plt.show()
    
    def export_ensemble_report(self) -> pd.DataFrame:
        """
        Exporta relatório detalhado do ensemble.
        
        Returns:
            pd.DataFrame: Relatório completo
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
            
            # Adicionar métricas de estabilidade
            stability = profile.get('stability', {})
            row['mean_stability'] = stability.get('mean_stability', 0)
            row['stable_samples'] = stability.get('stable_samples', 0)
            row['unstable_samples'] = stability.get('unstable_samples', 0)
            
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
    
    # Simular dados complexos de usuários gaming/apostas
    n_users = 1200
    
    # Diferentes tipos de usuários
    casual_users = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(1.5, 0.8, n_users//4),
        'total_deposits': np.random.lognormal(3, 1, n_users//4),
        'session_frequency': np.random.poisson(5, n_users//4),
        'avg_session_duration': np.random.exponential(20, n_users//4),
        'games_played': np.random.poisson(3, n_users//4),
        'preferred_hour': np.random.normal(19, 4, n_users//4),
        'days_since_last_bet': np.random.exponential(5, n_users//4),
        'win_rate': np.random.beta(1, 3, n_users//4),
        'cashback_usage': np.random.binomial(1, 0.2, n_users//4),
        'sports_bet_ratio': np.random.beta(1, 2, n_users//4)
    })
    
    regular_users = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(2.5, 1, n_users//3),
        'total_deposits': np.random.lognormal(4.5, 1.3, n_users//3),
        'session_frequency': np.random.poisson(12, n_users//3),
        'avg_session_duration': np.random.exponential(40, n_users//3),
        'games_played': np.random.poisson(6, n_users//3),
        'preferred_hour': np.random.normal(21, 3, n_users//3),
        'days_since_last_bet': np.random.exponential(2, n_users//3),
        'win_rate': np.random.beta(2, 3, n_users//3),
        'cashback_usage': np.random.binomial(1, 0.5, n_users//3),
        'sports_bet_ratio': np.random.beta(1.5, 1.5, n_users//3)
    })
    
    vip_users = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(4, 1.5, n_users//6),
        'total_deposits': np.random.lognormal(6.5, 2, n_users//6),
        'session_frequency': np.random.poisson(25, n_users//6),
        'avg_session_duration': np.random.exponential(80, n_users//6),
        'games_played': np.random.poisson(12, n_users//6),
        'preferred_hour': np.random.normal(23, 2, n_users//6),
        'days_since_last_bet': np.random.exponential(0.8, n_users//6),
        'win_rate': np.random.beta(3, 2, n_users//6),
        'cashback_usage': np.random.binomial(1, 0.8, n_users//6),
        'sports_bet_ratio': np.random.beta(2, 1, n_users//6)
    })
    
    # Outliers
    outliers = pd.DataFrame({
        'avg_bet_amount': np.random.lognormal(6, 3, n_users//4),
        'total_deposits': np.random.lognormal(8, 4, n_users//4),
        'session_frequency': np.random.poisson(100, n_users//4),
        'avg_session_duration': np.random.exponential(500, n_users//4),
        'games_played': np.random.poisson(50, n_users//4),
        'preferred_hour': np.random.uniform(0, 24, n_users//4),
        'days_since_last_bet': np.random.exponential(0.1, n_users//4),
        'win_rate': np.random.uniform(0, 1, n_users//4),
        'cashback_usage': np.random.binomial(1, 0.95, n_users//4),
        'sports_bet_ratio': np.random.uniform(0, 1, n_users//4)
    })
    
    # Combinar dados
    test_data = pd.concat([casual_users, regular_users, vip_users, outliers], ignore_index=True)
    
    # Testar Ensemble Clusterer
    print("🚀 Testando Ensemble Clusterer...")
    
    ensemble = EnsembleClusterer(
        use_kmeans=True,
        use_dbscan=True,
        use_hdbscan=HDBSCAN_AVAILABLE,
        weight_strategy='quality_based',
        final_clustering_method='weighted_voting'
    )
    
    ensemble.fit(test_data)
    
    metrics = ensemble.get_metrics()
    print(f"✅ Clusters do ensemble: {metrics['n_clusters']}")
    print(f"✅ Outliers detectados: {metrics['n_outliers']} ({metrics['outlier_ratio']*100:.1f}%)")
    print(f"✅ Estabilidade geral: {metrics['overall_stability']:.3f}")
    
    if 'silhouette_score' in metrics:
        print(f"✅ Silhouette Score: {metrics['silhouette_score']:.3f}")
    
    print(f"\n✅ Pesos dos algoritmos:")
    for algo, weight in metrics['algorithm_weights'].items():
        print(f"   {algo}: {weight:.3f}")
    
    print(f"\n✅ Perfis dos clusters do ensemble:")
    for cluster_id, profile in ensemble.get_cluster_profiles().items():
        stability_info = ""
        if 'mean_stability' in profile.get('stability', {}):
            stability_info = f" (estabilidade: {profile['stability']['mean_stability']:.3f})"
        print(f"   {profile['name']}: {profile['size']} usuários ({profile['percentage']:.1f}%){stability_info}")
    
    # Teste de predição
    new_users = test_data[:30]
    predictions = ensemble.predict(new_users)
    unique_preds, counts = np.unique(predictions, return_counts=True)
    print(f"\n✅ Predições para 30 novos usuários: clusters {unique_preds} com contagens {counts}")
    
    print("\n🎯 Ensemble Clusterer implementado com sucesso!")