#!/usr/bin/env python3
"""
ML Cluster - Script principal do sistema de clusterização gaming/apostas
Pipeline completo de ML para segmentação inteligente de usuários CRM.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM

Uso:
    python ml_cluster.py --mode train --input data.csv --output model.pkl
    python ml_cluster.py --mode predict --model model.pkl --input new_data.csv
    python ml_cluster.py --mode evaluate --model model.pkl --input data.csv
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

# Importar componentes do sistema
from models.cluster_model import GamingClusterModel
from utils.model_evaluation import ClusteringEvaluator
from utils.cluster_interpretation import ClusterInterpreter

warnings.filterwarnings('ignore')

# Configurar logging
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Configura sistema de logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

logger = logging.getLogger(__name__)

class MLClusterPipeline:
    """
    Pipeline principal de clusterização ML para gaming/apostas.
    
    Integra todos os componentes do sistema em um workflow completo.
    """
    
    def __init__(self, 
                 algorithm: str = 'ensemble',
                 feature_engines: List[str] = None,
                 output_dir: str = 'output',
                 random_state: int = 42):
        """
        Inicializa pipeline de ML.
        
        Args:
            algorithm: Algoritmo de clustering ('kmeans', 'dbscan', 'hdbscan', 'ensemble')
            feature_engines: Lista de engines de features
            output_dir: Diretório para outputs
            random_state: Seed para reprodutibilidade
        """
        self.algorithm = algorithm
        self.feature_engines = feature_engines or ['behavioral', 'temporal', 'financial', 'communication']
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Criar diretório de output
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Componentes do pipeline
        self.model = None
        self.evaluator = ClusteringEvaluator()
        self.interpreter = ClusterInterpreter()
        
        # Resultados
        self.results = {}
        
    def train_model(self, 
                   data_path: str,
                   interaction_data_path: Optional[str] = None,
                   user_id_col: str = 'user_id',
                   validation_split: float = 0.2,
                   save_model: bool = True) -> Dict[str, Any]:
        """
        Treina modelo de clusterização completo.
        
        Args:
            data_path: Caminho para dados principais
            interaction_data_path: Caminho para dados de interação (opcional)
            user_id_col: Nome da coluna de user_id
            validation_split: Proporção para validação
            save_model: Se deve salvar o modelo treinado
            
        Returns:
            Dict: Resultados do treinamento
        """
        logger.info("🚀 Iniciando treinamento do modelo de clusterização...")
        logger.info(f"Algoritmo: {self.algorithm}")
        logger.info(f"Feature engines: {self.feature_engines}")
        
        # Carregar dados
        logger.info(f"Carregando dados de: {data_path}")
        df = pd.read_csv(data_path)
        
        interaction_df = None
        if interaction_data_path:
            logger.info(f"Carregando dados de interação de: {interaction_data_path}")
            interaction_df = pd.read_csv(interaction_data_path)
        
        logger.info(f"Dataset principal: {df.shape}")
        if interaction_df is not None:
            logger.info(f"Dataset de interação: {interaction_df.shape}")
        
        # Inicializar modelo
        self.model = GamingClusterModel(
            clustering_algorithm=self.algorithm,
            feature_engines=self.feature_engines,
            random_state=self.random_state
        )
        
        # Treinar
        start_time = datetime.now()
        self.model.fit(
            df=df,
            user_id_col=user_id_col,
            interaction_data=interaction_df,
            validation_split=validation_split
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Avaliar modelo
        logger.info("📊 Avaliando qualidade do clustering...")
        evaluation = self.evaluator.evaluate_clustering(
            X=self.model.feature_matrix,
            labels=self.model.cluster_labels,
            algorithm_name=self.algorithm
        )
        
        # Interpretar clusters
        logger.info("🎯 Interpretando clusters para insights de negócio...")
        interpretation = self.interpreter.interpret_clusters(
            cluster_profiles=self.model.cluster_profiles,
            feature_matrix=self.model.feature_matrix,
            cluster_labels=self.model.cluster_labels,
            feature_importance=self.model.feature_importance
        )
        
        # Compilar resultados
        results = {
            'training_info': {
                'algorithm': self.algorithm,
                'feature_engines': self.feature_engines,
                'training_time_seconds': training_time,
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape,
                'interaction_data_available': interaction_df is not None
            },
            'model_summary': self.model.get_cluster_summary(),
            'evaluation': evaluation,
            'interpretation': interpretation,
            'feature_importance': self.model.feature_importance
        }
        
        # Salvar resultados
        self._save_training_results(results)
        
        # Salvar modelo se solicitado
        if save_model:
            model_path = self.output_dir / f"gaming_cluster_model_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            saved_path = self.model.save_model(str(model_path))
            results['model_path'] = saved_path
            logger.info(f"✅ Modelo salvo em: {saved_path}")
        
        # Gerar visualizações
        self._generate_visualizations(df)
        
        # Exportar dados para campanhas
        campaign_data_path = self._export_campaign_data()
        results['campaign_data_path'] = campaign_data_path
        
        logger.info("✅ Treinamento concluído com sucesso!")
        
        # Log resumo
        model_info = results['model_summary']['model_info']
        logger.info(f"📈 Resumo: {model_info['n_clusters']} clusters, {model_info['total_users']} usuários")
        
        if 'silhouette_score' in evaluation['intrinsic_metrics']:
            logger.info(f"📊 Silhouette Score: {evaluation['intrinsic_metrics']['silhouette_score']:.3f}")
        
        self.results = results
        return results
    
    def predict_clusters(self, 
                        model_path: str,
                        data_path: str,
                        interaction_data_path: Optional[str] = None,
                        user_id_col: str = 'user_id',
                        output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Prediz clusters para novos dados.
        
        Args:
            model_path: Caminho do modelo treinado
            data_path: Caminho para dados novos
            interaction_data_path: Dados de interação (opcional)
            user_id_col: Nome da coluna de user_id
            output_file: Arquivo para salvar predições
            
        Returns:
            pd.DataFrame: Dados com predições
        """
        logger.info("🔮 Iniciando predição de clusters...")
        
        # Carregar modelo
        logger.info(f"Carregando modelo de: {model_path}")
        self.model = GamingClusterModel.load_model(model_path)
        
        # Carregar dados
        logger.info(f"Carregando dados de: {data_path}")
        df = pd.read_csv(data_path)
        
        interaction_df = None
        if interaction_data_path:
            interaction_df = pd.read_csv(interaction_data_path)
        
        # Predizer
        logger.info(f"Predizindo clusters para {len(df)} usuários...")
        predictions = self.model.predict(
            df=df,
            user_id_col=user_id_col,
            interaction_data=interaction_df
        )
        
        # Adicionar predições aos dados
        if user_id_col in df.columns:
            result_df = df[[user_id_col]].copy()
        else:
            result_df = pd.DataFrame({'user_id': range(len(df))})
        
        result_df['cluster_id'] = predictions
        
        # Adicionar informações dos clusters
        for i, cluster_id in enumerate(predictions):
            if cluster_id in self.model.cluster_profiles:
                profile = self.model.cluster_profiles[cluster_id]
                result_df.loc[i, 'cluster_name'] = profile.get('name', f'Cluster_{cluster_id}')
                result_df.loc[i, 'is_outlier'] = cluster_id == -1
            else:
                result_df.loc[i, 'cluster_name'] = f'Cluster_{cluster_id}'
                result_df.loc[i, 'is_outlier'] = cluster_id == -1
        
        # Adicionar estatísticas
        unique_clusters, counts = np.unique(predictions, return_counts=True)
        logger.info(f"✅ Predições concluídas:")
        for cluster_id, count in zip(unique_clusters, counts):
            percentage = count / len(predictions) * 100
            cluster_name = result_df[result_df['cluster_id'] == cluster_id]['cluster_name'].iloc[0]
            logger.info(f"   {cluster_name}: {count} usuários ({percentage:.1f}%)")
        
        # Salvar se solicitado
        if output_file:
            output_path = self.output_dir / output_file
            result_df.to_csv(output_path, index=False)
            logger.info(f"✅ Predições salvas em: {output_path}")
        
        return result_df
    
    def evaluate_model(self, 
                      model_path: str,
                      data_path: str,
                      interaction_data_path: Optional[str] = None,
                      user_id_col: str = 'user_id',
                      generate_report: bool = True) -> Dict[str, Any]:
        """
        Avalia modelo existente.
        
        Args:
            model_path: Caminho do modelo
            data_path: Dados para avaliação
            interaction_data_path: Dados de interação (opcional)
            user_id_col: Nome da coluna de user_id
            generate_report: Se deve gerar relatório completo
            
        Returns:
            Dict: Resultados da avaliação
        """
        logger.info("📊 Iniciando avaliação do modelo...")
        
        # Carregar modelo
        self.model = GamingClusterModel.load_model(model_path)
        
        # Carregar dados
        df = pd.read_csv(data_path)
        interaction_df = None
        if interaction_data_path:
            interaction_df = pd.read_csv(interaction_data_path)
        
        # Predizer para obter labels
        predictions = self.model.predict(df, user_id_col, interaction_df)
        
        # Avaliar
        evaluation = self.evaluator.evaluate_clustering(
            X=self.model.feature_matrix,
            labels=predictions,
            algorithm_name=self.model.clustering_algorithm
        )
        
        logger.info("✅ Avaliação concluída:")
        
        intrinsic = evaluation['intrinsic_metrics']
        if 'silhouette_score' in intrinsic:
            logger.info(f"   Silhouette Score: {intrinsic['silhouette_score']:.3f}")
        if 'davies_bouldin_score' in intrinsic:
            logger.info(f"   Davies-Bouldin Score: {intrinsic['davies_bouldin_score']:.3f}")
        
        data_info = evaluation['data_info']
        logger.info(f"   Clusters: {data_info['n_clusters']}")
        logger.info(f"   Outliers: {data_info['n_outliers']} ({data_info['outlier_ratio']*100:.1f}%)")
        
        # Gerar relatório se solicitado
        if generate_report:
            report_path = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(evaluation, f, indent=2, default=str)
            logger.info(f"📄 Relatório salvo em: {report_path}")
        
        return evaluation
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Salva resultados do treinamento."""
        # Salvar resultados completos
        results_path = self.output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Salvar resumo executivo
        business_summary = self.interpreter.export_business_summary()
        summary_path = self.output_dir / f"business_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(business_summary, f, indent=2, default=str)
        
        logger.info(f"📄 Resultados salvos em: {results_path}")
        logger.info(f"📊 Resumo executivo salvo em: {summary_path}")
    
    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """Gera visualizações do clustering."""
        try:
            import matplotlib.pyplot as plt
            
            # Visualização do clustering
            viz_path = self.output_dir / f"clustering_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            self.evaluator.visualize_clustering(
                X=self.model.feature_matrix,
                labels=self.model.cluster_labels,
                algorithm_name=self.algorithm,
                save_path=str(viz_path)
            )
            
            logger.info(f"📈 Visualização salva em: {viz_path}")
            
        except Exception as e:
            logger.warning(f"Erro ao gerar visualizações: {e}")
    
    def _export_campaign_data(self) -> str:
        """Exporta dados formatados para campanhas."""
        campaign_data = self.model.export_for_campaigns()
        
        # Adicionar interpretações
        for i, row in campaign_data.iterrows():
            cluster_id = row['cluster_id']
            if cluster_id in self.interpreter.cluster_personas:
                persona = self.interpreter.cluster_personas[cluster_id]
                campaign_data.loc[i, 'archetype'] = persona.get('archetype', '')
                
                # Adicionar estratégia de campanha
                if cluster_id in self.interpreter.cluster_strategies:
                    strategy = self.interpreter.cluster_strategies[cluster_id]
                    campaign_strategy = strategy.get('campaign_strategy', {})
                    campaign_data.loc[i, 'recommended_campaign'] = campaign_strategy.get('name', '')
                    campaign_data.loc[i, 'optimal_frequency'] = campaign_strategy.get('frequency', '')
        
        # Salvar
        campaign_path = self.output_dir / f"campaign_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        campaign_data.to_csv(campaign_path, index=False)
        
        logger.info(f"📧 Dados para campanhas exportados: {campaign_path}")
        logger.info(f"   {len(campaign_data)} usuários prontos para segmentação")
        
        return str(campaign_path)
    
    def compare_algorithms(self, 
                          data_path: str,
                          algorithms: List[str] = None,
                          interaction_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compara múltiplos algoritmos de clustering.
        
        Args:
            data_path: Caminho dos dados
            algorithms: Lista de algoritmos para comparar
            interaction_data_path: Dados de interação (opcional)
            
        Returns:
            Dict: Comparação dos algoritmos
        """
        if algorithms is None:
            algorithms = ['kmeans', 'dbscan', 'ensemble']
        
        logger.info(f"🔬 Comparando algoritmos: {algorithms}")
        
        # Carregar dados
        df = pd.read_csv(data_path)
        interaction_df = None
        if interaction_data_path:
            interaction_df = pd.read_csv(interaction_data_path)
        
        evaluations = {}
        
        for algo in algorithms:
            logger.info(f"Testando {algo}...")
            
            try:
                # Treinar modelo
                model = GamingClusterModel(
                    clustering_algorithm=algo,
                    feature_engines=self.feature_engines,
                    random_state=self.random_state
                )
                
                model.fit(df, interaction_data=interaction_df, validation_split=0)
                
                # Avaliar
                evaluation = self.evaluator.evaluate_clustering(
                    X=model.feature_matrix,
                    labels=model.cluster_labels,
                    algorithm_name=algo
                )
                
                evaluations[algo] = evaluation
                
            except Exception as e:
                logger.error(f"Erro ao testar {algo}: {e}")
                continue
        
        # Comparar resultados
        if len(evaluations) > 1:
            comparison = self.evaluator.compare_algorithms(evaluations)
            
            # Salvar comparação
            comparison_path = self.output_dir / f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(comparison_path, 'w') as f:
                json.dump({'evaluations': evaluations, 'comparison': comparison}, f, indent=2, default=str)
            
            logger.info(f"✅ Comparação concluída:")
            logger.info(f"   Melhor algoritmo: {comparison['best_algorithm']}")
            logger.info(f"   Relatório salvo em: {comparison_path}")
            
            return comparison
        else:
            logger.warning("Não foi possível comparar algoritmos")
            return {}

def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(description='Sistema de Clusterização Gaming/Apostas CRM')
    
    # Argumentos principais
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'compare'], 
                       required=True, help='Modo de operação')
    parser.add_argument('--input', required=True, help='Arquivo de dados de entrada')
    parser.add_argument('--output', help='Arquivo/diretório de saída')
    parser.add_argument('--model', help='Caminho do modelo (para predict/evaluate)')
    
    # Configurações do modelo
    parser.add_argument('--algorithm', default='ensemble', 
                       choices=['kmeans', 'dbscan', 'hdbscan', 'ensemble'],
                       help='Algoritmo de clustering')
    parser.add_argument('--features', nargs='+', 
                       default=['behavioral', 'temporal', 'financial', 'communication'],
                       help='Feature engines a usar')
    
    # Dados adicionais
    parser.add_argument('--interaction-data', help='Arquivo de dados de interação')
    parser.add_argument('--user-id-col', default='user_id', help='Nome da coluna de user ID')
    
    # Configurações de execução
    parser.add_argument('--validation-split', type=float, default=0.2, 
                       help='Proporção para validação')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--output-dir', default='output', help='Diretório de saída')
    
    # Logging
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nível de logging')
    parser.add_argument('--log-file', help='Arquivo de log')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Verificar arquivo de entrada
    if not os.path.exists(args.input):
        logger.error(f"Arquivo de entrada não encontrado: {args.input}")
        sys.exit(1)
    
    # Inicializar pipeline
    pipeline = MLClusterPipeline(
        algorithm=args.algorithm,
        feature_engines=args.features,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    try:
        if args.mode == 'train':
            logger.info("🚀 Modo: Treinamento")
            results = pipeline.train_model(
                data_path=args.input,
                interaction_data_path=args.interaction_data,
                user_id_col=args.user_id_col,
                validation_split=args.validation_split,
                save_model=True
            )
            
            print("\n" + "="*50)
            print("🎯 TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print("="*50)
            
            model_info = results['model_summary']['model_info']
            print(f"📊 Clusters gerados: {model_info['n_clusters']}")
            print(f"👥 Usuários processados: {model_info['total_users']}")
            print(f"🎪 Outliers detectados: {model_info['outliers']}")
            
            if 'model_path' in results:
                print(f"💾 Modelo salvo: {results['model_path']}")
            
            if 'campaign_data_path' in results:
                print(f"📧 Dados de campanha: {results['campaign_data_path']}")
        
        elif args.mode == 'predict':
            logger.info("🔮 Modo: Predição")
            if not args.model:
                logger.error("Modelo é obrigatório para predição")
                sys.exit(1)
            
            predictions = pipeline.predict_clusters(
                model_path=args.model,
                data_path=args.input,
                interaction_data_path=args.interaction_data,
                user_id_col=args.user_id_col,
                output_file=args.output
            )
            
            print("\n" + "="*50)
            print("🔮 PREDIÇÃO CONCLUÍDA COM SUCESSO!")
            print("="*50)
            print(f"📊 Usuários processados: {len(predictions)}")
            print(f"🎯 Clusters únicos: {predictions['cluster_id'].nunique()}")
        
        elif args.mode == 'evaluate':
            logger.info("📊 Modo: Avaliação")
            if not args.model:
                logger.error("Modelo é obrigatório para avaliação")
                sys.exit(1)
            
            evaluation = pipeline.evaluate_model(
                model_path=args.model,
                data_path=args.input,
                interaction_data_path=args.interaction_data,
                user_id_col=args.user_id_col
            )
            
            print("\n" + "="*50)
            print("📊 AVALIAÇÃO CONCLUÍDA!")
            print("="*50)
        
        elif args.mode == 'compare':
            logger.info("🔬 Modo: Comparação")
            algorithms = ['kmeans', 'dbscan', 'ensemble']
            
            comparison = pipeline.compare_algorithms(
                data_path=args.input,
                algorithms=algorithms,
                interaction_data_path=args.interaction_data
            )
            
            if comparison:
                print("\n" + "="*50)
                print("🔬 COMPARAÇÃO CONCLUÍDA!")
                print("="*50)
                print(f"🏆 Melhor algoritmo: {comparison['best_algorithm']}")
    
    except Exception as e:
        logger.error(f"Erro durante execução: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()