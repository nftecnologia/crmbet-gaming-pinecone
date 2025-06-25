"""
Testes unitários para algoritmos de clustering do sistema gaming/apostas.
Testa funcionalidades básicas e casos limite dos algoritmos implementados.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from clustering.kmeans_clusterer import KMeansClusterer
from clustering.dbscan_clusterer import DBSCANClusterer
from clustering.ensemble_clusterer import EnsembleClusterer

# Tentar importar HDBSCAN
try:
    from clustering.hdbscan_clusterer import HDBSCANClusterer
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

class TestKMeansClusterer:
    """Testes para KMeansClusterer."""
    
    @pytest.fixture
    def sample_data(self):
        """Gera dados de teste."""
        np.random.seed(42)
        return pd.DataFrame({
            'avg_bet_amount': np.random.lognormal(2, 1, 100),
            'session_frequency': np.random.poisson(10, 100),
            'win_rate': np.random.beta(2, 3, 100),
            'session_duration': np.random.exponential(30, 100)
        })
    
    def test_initialization(self):
        """Testa inicialização do KMeansClusterer."""
        clusterer = KMeansClusterer()
        assert clusterer.n_clusters_range == (3, 12)
        assert clusterer.scaler_type == 'standard'
        assert clusterer.random_state == 42
    
    def test_fit_basic(self, sample_data):
        """Testa fit básico."""
        clusterer = KMeansClusterer(n_clusters_range=(2, 5))
        clusterer.fit(sample_data)
        
        assert clusterer.model is not None
        assert clusterer.optimal_k >= 2
        assert clusterer.optimal_k <= 5
        assert len(clusterer.cluster_profiles) > 0
    
    def test_predict(self, sample_data):
        """Testa predição."""
        clusterer = KMeansClusterer(n_clusters_range=(3, 4))
        clusterer.fit(sample_data)
        
        # Predizer nos mesmos dados
        predictions = clusterer.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert len(np.unique(predictions)) >= 2
    
    def test_feature_importance(self, sample_data):
        """Testa cálculo de importância das features."""
        clusterer = KMeansClusterer()
        clusterer.fit(sample_data)
        
        importance = clusterer.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == len(sample_data.columns)
        
        # Verificar se estão ordenadas por importância
        importance_values = list(importance.values())
        assert importance_values == sorted(importance_values, reverse=True)
    
    def test_cluster_profiles(self, sample_data):
        """Testa geração de perfis de cluster."""
        clusterer = KMeansClusterer()
        clusterer.fit(sample_data)
        
        profiles = clusterer.get_cluster_profiles()
        assert isinstance(profiles, dict)
        
        for cluster_id, profile in profiles.items():
            assert 'size' in profile
            assert 'percentage' in profile
            assert 'name' in profile
            assert 'statistics' in profile
    
    def test_export_report(self, sample_data):
        """Testa exportação de relatório."""
        clusterer = KMeansClusterer()
        clusterer.fit(sample_data)
        
        report = clusterer.export_cluster_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
        assert 'cluster_id' in report.columns
        assert 'cluster_name' in report.columns
    
    def test_empty_data(self):
        """Testa comportamento com dados vazios."""
        clusterer = KMeansClusterer()
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError):
            clusterer.fit(empty_data)
    
    def test_small_dataset(self):
        """Testa comportamento com dataset muito pequeno."""
        clusterer = KMeansClusterer()
        small_data = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        })
        
        with pytest.raises(ValueError):
            clusterer.fit(small_data)

class TestDBSCANClusterer:
    """Testes para DBSCANClusterer."""
    
    @pytest.fixture
    def sample_data_with_outliers(self):
        """Gera dados com outliers para DBSCAN."""
        np.random.seed(42)
        
        # Dados normais
        normal_data = pd.DataFrame({
            'avg_bet_amount': np.random.lognormal(2, 0.5, 80),
            'session_frequency': np.random.poisson(8, 80),
            'win_rate': np.random.beta(2, 3, 80)
        })
        
        # Outliers
        outlier_data = pd.DataFrame({
            'avg_bet_amount': np.random.lognormal(5, 2, 20),
            'session_frequency': np.random.poisson(50, 20),
            'win_rate': np.random.uniform(0, 1, 20)
        })
        
        return pd.concat([normal_data, outlier_data], ignore_index=True)
    
    def test_initialization(self):
        """Testa inicialização do DBSCANClusterer."""
        clusterer = DBSCANClusterer()
        assert clusterer.eps_range == (0.1, 2.0)
        assert clusterer.min_samples_range == (5, 50)
        assert clusterer.metric == 'euclidean'
    
    def test_fit_basic(self, sample_data_with_outliers):
        """Testa fit básico do DBSCAN."""
        clusterer = DBSCANClusterer()
        clusterer.fit(sample_data_with_outliers)
        
        assert clusterer.model is not None
        assert clusterer.optimal_eps > 0
        assert clusterer.optimal_min_samples > 0
    
    def test_outlier_detection(self, sample_data_with_outliers):
        """Testa detecção de outliers."""
        clusterer = DBSCANClusterer()
        clusterer.fit(sample_data_with_outliers)
        
        outlier_analysis = clusterer.get_outlier_analysis()
        assert isinstance(outlier_analysis, dict)
        assert 'count' in outlier_analysis
        assert 'percentage' in outlier_analysis
        
        # DBSCAN deve detectar alguns outliers
        assert outlier_analysis['count'] > 0
    
    def test_get_outliers_dataframe(self, sample_data_with_outliers):
        """Testa obtenção de DataFrame de outliers."""
        clusterer = DBSCANClusterer()
        clusterer.fit(sample_data_with_outliers)
        
        outliers_df = clusterer.get_outliers(sample_data_with_outliers)
        assert isinstance(outliers_df, pd.DataFrame)
        
        if len(outliers_df) > 0:
            assert 'cluster' in outliers_df.columns
            assert 'outlier_score' in outliers_df.columns
            assert all(outliers_df['cluster'] == -1)

class TestEnsembleClusterer:
    """Testes para EnsembleClusterer."""
    
    @pytest.fixture
    def sample_data(self):
        """Gera dados de teste para ensemble."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(150),
            'feature2': np.random.randn(150),
            'feature3': np.random.randn(150),
            'feature4': np.random.randn(150)
        })
    
    def test_initialization_default(self):
        """Testa inicialização padrão do EnsembleClusterer."""
        clusterer = EnsembleClusterer()
        assert clusterer.use_kmeans == True
        assert clusterer.use_dbscan == True
        assert clusterer.consensus_threshold == 0.6
    
    def test_initialization_custom(self):
        """Testa inicialização customizada."""
        clusterer = EnsembleClusterer(
            use_kmeans=True,
            use_dbscan=False,
            use_hdbscan=False,
            consensus_threshold=0.8
        )
        assert clusterer.use_kmeans == True
        assert clusterer.use_dbscan == False
        assert clusterer.use_hdbscan == False
        assert clusterer.consensus_threshold == 0.8
    
    def test_fit_ensemble(self, sample_data):
        """Testa fit do ensemble."""
        clusterer = EnsembleClusterer(
            use_kmeans=True,
            use_dbscan=True,
            use_hdbscan=False  # Desabilitar para não depender de HDBSCAN
        )
        clusterer.fit(sample_data)
        
        assert clusterer.final_labels is not None
        assert len(clusterer.final_labels) == len(sample_data)
        assert len(clusterer.individual_results) >= 2  # KMeans + DBSCAN
    
    def test_individual_weights(self, sample_data):
        """Testa cálculo de pesos individuais."""
        clusterer = EnsembleClusterer(weight_strategy='quality_based')
        clusterer.fit(sample_data)
        
        weights = clusterer.individual_weights
        assert isinstance(weights, dict)
        assert len(weights) >= 2
        
        # Pesos devem somar aproximadamente 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.1
    
    def test_stability_analysis(self, sample_data):
        """Testa análise de estabilidade."""
        clusterer = EnsembleClusterer()
        clusterer.fit(sample_data)
        
        stability = clusterer.get_stability_analysis()
        assert isinstance(stability, dict)
        assert 'overall_stability' in stability
        assert 'cluster_stability' in stability
    
    def test_predict_ensemble(self, sample_data):
        """Testa predição do ensemble."""
        clusterer = EnsembleClusterer()
        clusterer.fit(sample_data)
        
        # Predizer nos mesmos dados
        predictions = clusterer.predict(sample_data)
        assert len(predictions) == len(sample_data)
    
    def test_export_report(self, sample_data):
        """Testa exportação de relatório do ensemble."""
        clusterer = EnsembleClusterer()
        clusterer.fit(sample_data)
        
        report = clusterer.export_ensemble_report()
        assert isinstance(report, pd.DataFrame)
        assert 'cluster_id' in report.columns
        assert 'mean_stability' in report.columns

@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="HDBSCAN não disponível")
class TestHDBSCANClusterer:
    """Testes para HDBSCANClusterer (apenas se disponível)."""
    
    @pytest.fixture
    def sample_data(self):
        """Gera dados hierárquicos para HDBSCAN."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
    
    def test_initialization(self):
        """Testa inicialização do HDBSCANClusterer."""
        clusterer = HDBSCANClusterer()
        assert clusterer.min_cluster_size_range == (5, 100)
        assert clusterer.cluster_selection_method == 'eom'
    
    def test_fit_basic(self, sample_data):
        """Testa fit básico do HDBSCAN."""
        clusterer = HDBSCANClusterer()
        clusterer.fit(sample_data)
        
        assert clusterer.model is not None
        assert clusterer.optimal_min_cluster_size > 0
    
    def test_hierarchy_analysis(self, sample_data):
        """Testa análise hierárquica."""
        clusterer = HDBSCANClusterer()
        clusterer.fit(sample_data)
        
        hierarchy = clusterer.get_hierarchy_analysis()
        assert isinstance(hierarchy, dict)
        assert 'has_hierarchy' in hierarchy
    
    def test_predict_with_probabilities(self, sample_data):
        """Testa predição com probabilidades."""
        clusterer = HDBSCANClusterer(prediction_data=True)
        clusterer.fit(sample_data)
        
        predictions = clusterer.predict(sample_data.head(10))
        probabilities = clusterer.predict_proba(sample_data.head(10))
        
        assert len(predictions) == 10
        assert len(probabilities) == 10

class TestIntegration:
    """Testes de integração entre componentes."""
    
    @pytest.fixture
    def gaming_data(self):
        """Gera dados realísticos de gaming."""
        np.random.seed(42)
        n_users = 200
        
        return pd.DataFrame({
            'avg_bet_amount': np.random.lognormal(3, 1.5, n_users),
            'session_frequency': np.random.poisson(12, n_users),
            'win_rate': np.random.beta(2, 3, n_users),
            'session_duration': np.random.exponential(45, n_users),
            'games_played': np.random.poisson(6, n_users),
            'days_since_last_bet': np.random.exponential(3, n_users),
            'total_deposits': np.random.lognormal(5, 2, n_users)
        })
    
    def test_all_algorithms_consistency(self, gaming_data):
        """Testa se todos os algoritmos produzem resultados consistentes."""
        algorithms = [
            ('kmeans', KMeansClusterer()),
            ('dbscan', DBSCANClusterer())
        ]
        
        if HDBSCAN_AVAILABLE:
            algorithms.append(('hdbscan', HDBSCANClusterer()))
        
        results = {}
        
        for name, clusterer in algorithms:
            try:
                clusterer.fit(gaming_data)
                results[name] = {
                    'n_clusters': len(clusterer.get_cluster_profiles()),
                    'has_outliers': -1 in [k for k in clusterer.get_cluster_profiles().keys()]
                }
            except Exception as e:
                pytest.fail(f"Algoritmo {name} falhou: {e}")
        
        # Verificar se todos produziram resultados válidos
        assert len(results) >= 2
        for name, result in results.items():
            assert result['n_clusters'] > 0, f"{name} não produziu clusters válidos"
    
    def test_ensemble_with_all_algorithms(self, gaming_data):
        """Testa ensemble com todos os algoritmos disponíveis."""
        clusterer = EnsembleClusterer(
            use_kmeans=True,
            use_dbscan=True,
            use_hdbscan=HDBSCAN_AVAILABLE
        )
        
        clusterer.fit(gaming_data)
        
        # Verificar resultados
        assert clusterer.final_labels is not None
        assert len(clusterer.individual_results) >= 2
        
        # Verificar métricas
        metrics = clusterer.get_metrics()
        assert 'n_clusters' in metrics
        assert 'algorithm_weights' in metrics
        
        # Verificar estabilidade
        stability = clusterer.get_stability_analysis()
        assert 'overall_stability' in stability

# Testes de performance (apenas informativos)
class TestPerformance:
    """Testes de performance para datasets maiores."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Testa performance com dataset grande."""
        np.random.seed(42)
        large_data = pd.DataFrame({
            'feature1': np.random.randn(5000),
            'feature2': np.random.randn(5000),
            'feature3': np.random.randn(5000),
            'feature4': np.random.randn(5000)
        })
        
        import time
        
        # Testar KMeans
        start_time = time.time()
        clusterer = KMeansClusterer()
        clusterer.fit(large_data)
        kmeans_time = time.time() - start_time
        
        # Testar DBSCAN
        start_time = time.time()
        clusterer = DBSCANClusterer()
        clusterer.fit(large_data)
        dbscan_time = time.time() - start_time
        
        print(f"\nPerformance com 5000 amostras:")
        print(f"KMeans: {kmeans_time:.2f}s")
        print(f"DBSCAN: {dbscan_time:.2f}s")
        
        # Verificar se tempos são razoáveis (< 60s)
        assert kmeans_time < 60, "KMeans muito lento"
        assert dbscan_time < 60, "DBSCAN muito lento"

if __name__ == "__main__":
    # Executar testes
    pytest.main([__file__, "-v"])