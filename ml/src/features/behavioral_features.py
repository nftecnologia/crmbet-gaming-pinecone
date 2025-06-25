"""
Behavioral Features Engineering - Extração de padrões comportamentais de usuários gaming/apostas
Implementação científica para identificar comportamentos únicos e características de jogo.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class BehavioralFeatureEngine:
    """
    Engine para extração de features comportamentais de usuários gaming/apostas.
    
    Features extraídas:
    - Padrões de risco e volatilidade
    - Consistência e regularidade 
    - Comportamento multi-jogo
    - Engagement e lealdade
    - Adaptabilidade e exploração
    """
    
    def __init__(self, 
                 risk_tolerance_bins: int = 5,
                 volatility_window: int = 7,
                 consistency_threshold: float = 0.3,
                 activity_percentiles: List[float] = [25, 50, 75, 90]):
        """
        Inicializa o engine de features comportamentais.
        
        Args:
            risk_tolerance_bins: Número de bins para categorização de risco
            volatility_window: Janela para calcular volatilidade
            consistency_threshold: Threshold para determinar consistência
            activity_percentiles: Percentis para análise de atividade
        """
        self.risk_tolerance_bins = risk_tolerance_bins
        self.volatility_window = volatility_window
        self.consistency_threshold = consistency_threshold
        self.activity_percentiles = activity_percentiles
        
        # Caches para otimização
        self._feature_cache = {}
        self._computed_features = set()
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai todas as features comportamentais.
        
        Args:
            df: DataFrame com dados de usuários e transações
            
        Returns:
            pd.DataFrame: DataFrame com features comportamentais
        """
        logger.info("Iniciando extração de features comportamentais...")
        
        # Validar dados de entrada
        self._validate_input_data(df)
        
        # DataFrame para features
        features_df = pd.DataFrame(index=df.index if 'user_id' not in df.columns else df['user_id'].unique())
        
        # Agrupar por usuário se necessário
        if 'user_id' in df.columns:
            user_data = df.groupby('user_id')
        else:
            # Assumir que cada linha é um usuário
            user_data = [(i, row) for i, row in df.iterrows()]
        
        # Extrair features por categoria
        features_df = self._extract_risk_features(user_data, features_df)
        features_df = self._extract_consistency_features(user_data, features_df)
        features_df = self._extract_engagement_features(user_data, features_df)
        features_df = self._extract_game_diversity_features(user_data, features_df)
        features_df = self._extract_learning_adaptation_features(user_data, features_df)
        features_df = self._extract_social_behavior_features(user_data, features_df)
        
        logger.info(f"Extração concluída. {len(features_df.columns)} features comportamentais criadas")
        return features_df
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Valida se os dados de entrada têm as colunas necessárias."""
        required_base_cols = ['bet_amount', 'session_duration', 'games_played']
        
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colunas recomendadas ausentes: {missing_cols}")
    
    def _extract_risk_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features relacionadas ao perfil de risco."""
        logger.info("Extraindo features de perfil de risco...")
        
        risk_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            # Dados agrupados por usuário
            for user_id, user_df in user_data:
                risk_features[user_id] = self._calculate_user_risk_profile(user_df)
        else:
            # Dados linha por linha
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    risk_features[idx] = self._calculate_single_user_risk_profile(user_row)
        
        # Converter para DataFrame
        risk_df = pd.DataFrame.from_dict(risk_features, orient='index')
        
        # Combinar com features_df
        for col in risk_df.columns:
            features_df[f'risk_{col}'] = risk_df[col]
        
        return features_df
    
    def _calculate_user_risk_profile(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula perfil de risco para usuário com múltiplas transações."""
        profile = {}
        
        if 'bet_amount' in user_df.columns:
            bets = user_df['bet_amount'].dropna()
            
            if len(bets) > 1:
                # Volatilidade das apostas
                profile['bet_volatility'] = bets.std() / (bets.mean() + 1e-8)
                
                # Escalação progressiva (tendência crescente)
                if len(bets) >= 3:
                    x = np.arange(len(bets))
                    slope, _, _, p_value, _ = stats.linregress(x, bets)
                    profile['bet_escalation'] = slope / (bets.mean() + 1e-8)
                    profile['escalation_significance'] = 1 - p_value
                else:
                    profile['bet_escalation'] = 0
                    profile['escalation_significance'] = 0
                
                # Análise de quantis (comportamento de cauda)
                profile['bet_p95_ratio'] = bets.quantile(0.95) / (bets.median() + 1e-8)
                profile['bet_max_ratio'] = bets.max() / (bets.mean() + 1e-8)
                
                # Impulsividade (mudanças bruscas)
                bet_changes = bets.diff().abs()
                profile['impulsivity_score'] = bet_changes.mean() / (bets.mean() + 1e-8)
                
                # Consistência de risco
                bet_cv = bets.std() / (bets.mean() + 1e-8)
                profile['risk_consistency'] = 1 / (1 + bet_cv)  # Invertido: maior consistência = menor CV
                
            else:
                # Valores padrão para casos com poucos dados
                profile.update({
                    'bet_volatility': 0,
                    'bet_escalation': 0,
                    'escalation_significance': 0,
                    'bet_p95_ratio': 1,
                    'bet_max_ratio': 1,
                    'impulsivity_score': 0,
                    'risk_consistency': 1
                })
        
        # Análise de win/loss behavior se disponível
        if 'win_amount' in user_df.columns and 'bet_amount' in user_df.columns:
            wins = user_df['win_amount'].fillna(0)
            bets = user_df['bet_amount'].dropna()
            
            if len(wins) > 0 and len(bets) > 0:
                # Chasing losses (aumentar apostas após perdas)
                losses = bets - wins
                loss_mask = losses > 0
                
                if loss_mask.sum() > 1:
                    subsequent_bets = bets.shift(-1)[loss_mask]
                    current_bets = bets[loss_mask]
                    
                    if len(subsequent_bets.dropna()) > 0:
                        chase_ratio = (subsequent_bets / current_bets).dropna()
                        profile['loss_chasing_tendency'] = (chase_ratio > 1.2).mean()
                    else:
                        profile['loss_chasing_tendency'] = 0
                else:
                    profile['loss_chasing_tendency'] = 0
        
        return profile
    
    def _calculate_single_user_risk_profile(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula perfil de risco para usuário com dados agregados."""
        profile = {}
        
        # Features baseadas em agregações existentes
        if 'avg_bet_amount' in user_row and 'max_bet_amount' in user_row:
            avg_bet = user_row.get('avg_bet_amount', 0)
            max_bet = user_row.get('max_bet_amount', 0)
            
            profile['bet_max_ratio'] = max_bet / (avg_bet + 1e-8) if avg_bet > 0 else 1
        
        if 'total_bets' in user_row and 'total_amount' in user_row:
            total_bets = user_row.get('total_bets', 0)
            total_amount = user_row.get('total_amount', 0)
            
            if total_bets > 0:
                avg_bet = total_amount / total_bets
                profile['avg_bet_normalized'] = avg_bet
        
        # Win rate como proxy para perfil de risco
        if 'win_rate' in user_row:
            win_rate = user_row.get('win_rate', 0.5)
            # Usuarios com win rate muito baixo ou muito alto podem ser mais arriscados
            profile['win_rate_extremity'] = abs(win_rate - 0.5) * 2
        
        # Frequência como indicador de compulsividade
        if 'session_frequency' in user_row:
            freq = user_row.get('session_frequency', 0)
            profile['session_frequency_normalized'] = min(freq / 30, 2)  # Normalizar por 30 sessões/mês
        
        return profile
    
    def _extract_consistency_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de consistência comportamental."""
        logger.info("Extraindo features de consistência...")
        
        consistency_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                consistency_features[user_id] = self._calculate_user_consistency(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    consistency_features[idx] = self._calculate_single_user_consistency(user_row)
        
        # Converter para DataFrame
        consistency_df = pd.DataFrame.from_dict(consistency_features, orient='index')
        
        # Combinar com features_df
        for col in consistency_df.columns:
            features_df[f'consistency_{col}'] = consistency_df[col]
        
        return features_df
    
    def _calculate_user_consistency(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de consistência para usuário."""
        profile = {}
        
        # Consistência temporal se temos timestamps
        if 'timestamp' in user_df.columns:
            timestamps = pd.to_datetime(user_df['timestamp'])
            time_diffs = timestamps.diff().dt.total_seconds() / 3600  # em horas
            
            if len(time_diffs.dropna()) > 1:
                # Regularidade das sessões
                profile['session_regularity'] = 1 / (1 + time_diffs.std() / (time_diffs.mean() + 1e-8))
                
                # Identificar padrões semanais
                weekdays = timestamps.dt.dayofweek
                profile['weekday_consistency'] = 1 - (weekdays.value_counts().std() / len(weekdays))
        
        # Consistência de comportamento de jogo
        if 'session_duration' in user_df.columns:
            durations = user_df['session_duration'].dropna()
            
            if len(durations) > 1:
                cv_duration = durations.std() / (durations.mean() + 1e-8)
                profile['duration_consistency'] = 1 / (1 + cv_duration)
        
        if 'games_played' in user_df.columns:
            games = user_df['games_played'].dropna()
            
            if len(games) > 1:
                cv_games = games.std() / (games.mean() + 1e-8)
                profile['games_consistency'] = 1 / (1 + cv_games)
        
        return profile
    
    def _calculate_single_user_consistency(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula consistência para dados agregados."""
        profile = {}
        
        # Proxy de consistência baseado em dados disponíveis
        if 'days_active' in user_row and 'total_sessions' in user_row:
            days_active = user_row.get('days_active', 1)
            total_sessions = user_row.get('total_sessions', 1)
            
            # Consistência baseada na distribuição de sessões
            sessions_per_day = total_sessions / max(days_active, 1)
            profile['daily_activity_consistency'] = min(sessions_per_day, 5) / 5  # Normalizar
        
        return profile
    
    def _extract_engagement_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de engajamento e lealdade."""
        logger.info("Extraindo features de engajamento...")
        
        engagement_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                engagement_features[user_id] = self._calculate_user_engagement(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    engagement_features[idx] = self._calculate_single_user_engagement(user_row)
        
        # Converter para DataFrame
        engagement_df = pd.DataFrame.from_dict(engagement_features, orient='index')
        
        # Combinar com features_df
        for col in engagement_df.columns:
            features_df[f'engagement_{col}'] = engagement_df[col]
        
        return features_df
    
    def _calculate_user_engagement(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de engajamento."""
        profile = {}
        
        # Intensidade das sessões
        if 'session_duration' in user_df.columns and 'games_played' in user_df.columns:
            durations = user_df['session_duration'].dropna()
            games = user_df['games_played'].dropna()
            
            if len(durations) > 0 and len(games) > 0:
                # Eficiência de tempo (jogos por minuto)
                total_time = durations.sum()
                total_games = games.sum()
                
                if total_time > 0:
                    profile['games_per_minute'] = total_games / (total_time / 60)
                
                # Intensidade da sessão média
                avg_duration = durations.mean()
                avg_games = games.mean()
                profile['session_intensity'] = avg_games / (avg_duration / 60 + 1e-8)
        
        # Evolução temporal do engajamento
        if 'timestamp' in user_df.columns:
            # Ordenar por timestamp
            user_df_sorted = user_df.sort_values('timestamp')
            
            # Calcular engajamento ao longo do tempo
            if 'session_duration' in user_df_sorted.columns:
                durations = user_df_sorted['session_duration'].rolling(window=3, min_periods=1).mean()
                
                if len(durations) >= 3:
                    # Tendência de engajamento (primeira vs última parte)
                    first_half = durations[:len(durations)//2].mean()
                    second_half = durations[len(durations)//2:].mean()
                    
                    profile['engagement_trend'] = (second_half - first_half) / (first_half + 1e-8)
        
        return profile
    
    def _calculate_single_user_engagement(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula engajamento para dados agregados."""
        profile = {}
        
        # Métricas de intensidade
        if 'avg_session_duration' in user_row and 'avg_games_per_session' in user_row:
            avg_duration = user_row.get('avg_session_duration', 0)
            avg_games = user_row.get('avg_games_per_session', 0)
            
            if avg_duration > 0:
                profile['session_intensity'] = avg_games / (avg_duration / 60)
        
        # Loyalty score baseado em atividade recente
        if 'days_since_last_bet' in user_row:
            days_since = user_row.get('days_since_last_bet', 0)
            profile['recency_score'] = 1 / (1 + days_since / 7)  # Decaimento semanal
        
        # Frequência de uso
        if 'session_frequency' in user_row:
            frequency = user_row.get('session_frequency', 0)
            profile['frequency_score'] = min(frequency / 30, 1)  # Normalizar por mês
        
        return profile
    
    def _extract_game_diversity_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de diversidade de jogos."""
        logger.info("Extraindo features de diversidade de jogos...")
        
        diversity_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                diversity_features[user_id] = self._calculate_game_diversity(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    diversity_features[idx] = self._calculate_single_user_diversity(user_row)
        
        # Converter para DataFrame
        diversity_df = pd.DataFrame.from_dict(diversity_features, orient='index')
        
        # Combinar com features_df
        for col in diversity_df.columns:
            features_df[f'diversity_{col}'] = diversity_df[col]
        
        return features_df
    
    def _calculate_game_diversity(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula diversidade de jogos."""
        profile = {}
        
        if 'game_type' in user_df.columns:
            game_types = user_df['game_type'].dropna()
            
            if len(game_types) > 0:
                # Shannon entropy para diversidade
                type_counts = game_types.value_counts()
                type_probs = type_counts / len(game_types)
                profile['game_entropy'] = entropy(type_probs)
                
                # Número de tipos únicos
                profile['unique_game_types'] = len(type_counts)
                
                # Concentração (inverso de diversidade)
                profile['game_concentration'] = (type_probs ** 2).sum()  # Herfindahl index
        
        # Exploração vs exploitação
        if 'game_type' in user_df.columns and 'timestamp' in user_df.columns:
            # Ordenar por tempo
            user_df_sorted = user_df.sort_values('timestamp')
            game_sequence = user_df_sorted['game_type'].tolist()
            
            # Calcular switching rate
            switches = sum(1 for i in range(1, len(game_sequence)) 
                          if game_sequence[i] != game_sequence[i-1])
            
            profile['game_switching_rate'] = switches / (len(game_sequence) - 1) if len(game_sequence) > 1 else 0
        
        return profile
    
    def _calculate_single_user_diversity(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula diversidade para dados agregados."""
        profile = {}
        
        # Proxy de diversidade baseado em ratios disponíveis
        game_ratios = []
        ratio_cols = [col for col in user_row.index if 'ratio' in col.lower() and 'game' in col.lower()]
        
        for col in ratio_cols:
            value = user_row.get(col, 0)
            if 0 <= value <= 1:
                game_ratios.append(value)
        
        if len(game_ratios) > 0:
            # Diversidade baseada na distribuição dos ratios
            game_ratios = np.array(game_ratios)
            # Normalizar para que somem 1
            if game_ratios.sum() > 0:
                game_ratios = game_ratios / game_ratios.sum()
                profile['ratio_entropy'] = entropy(game_ratios)
        
        # Sports betting ratio como indicador de especialização
        if 'sports_bet_ratio' in user_row:
            sports_ratio = user_row.get('sports_bet_ratio', 0.5)
            # Extremos indicam especialização
            profile['specialization_score'] = abs(sports_ratio - 0.5) * 2
        
        return profile
    
    def _extract_learning_adaptation_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de aprendizado e adaptação."""
        logger.info("Extraindo features de aprendizado e adaptação...")
        
        learning_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                learning_features[user_id] = self._calculate_learning_patterns(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    learning_features[idx] = self._calculate_single_user_learning(user_row)
        
        # Converter para DataFrame
        learning_df = pd.DataFrame.from_dict(learning_features, orient='index')
        
        # Combinar com features_df
        for col in learning_df.columns:
            features_df[f'learning_{col}'] = learning_df[col]
        
        return features_df
    
    def _calculate_learning_patterns(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula padrões de aprendizado."""
        profile = {}
        
        # Melhoria no desempenho ao longo do tempo
        if 'win_rate' in user_df.columns and 'timestamp' in user_df.columns:
            # Ordenar por tempo
            user_df_sorted = user_df.sort_values('timestamp')
            win_rates = user_df_sorted['win_rate'].dropna()
            
            if len(win_rates) >= 3:
                # Tendência de melhoria
                x = np.arange(len(win_rates))
                slope, _, r_value, p_value, _ = stats.linregress(x, win_rates)
                
                profile['performance_improvement'] = slope
                profile['improvement_consistency'] = r_value ** 2
                profile['improvement_significance'] = 1 - p_value
        
        # Adaptação de estratégia (mudanças no comportamento)
        if 'bet_amount' in user_df.columns and 'timestamp' in user_df.columns:
            user_df_sorted = user_df.sort_values('timestamp')
            bets = user_df_sorted['bet_amount'].dropna()
            
            if len(bets) >= 5:
                # Detectar mudanças de regime usando rolling statistics
                rolling_mean = bets.rolling(window=3, min_periods=1).mean()
                rolling_std = bets.rolling(window=3, min_periods=1).std()
                
                # Variação na estratégia
                strategy_changes = (rolling_std / (rolling_mean + 1e-8)).std()
                profile['strategy_adaptation'] = strategy_changes
        
        return profile
    
    def _calculate_single_user_learning(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula learning para dados agregados."""
        profile = {}
        
        # Win rate como proxy de skill
        if 'win_rate' in user_row:
            win_rate = user_row.get('win_rate', 0.5)
            # Win rates extremos podem indicar skill ou sorte
            profile['skill_indicator'] = abs(win_rate - 0.5)
        
        # Experiência baseada em volume
        if 'total_bets' in user_row:
            total_bets = user_row.get('total_bets', 0)
            profile['experience_level'] = min(np.log1p(total_bets) / 10, 1)  # Log-normalizado
        
        return profile
    
    def _extract_social_behavior_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de comportamento social."""
        logger.info("Extraindo features de comportamento social...")
        
        social_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                social_features[user_id] = self._calculate_social_behavior(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    social_features[idx] = self._calculate_single_user_social(user_row)
        
        # Converter para DataFrame
        social_df = pd.DataFrame.from_dict(social_features, orient='index')
        
        # Combinar com features_df
        for col in social_df.columns:
            features_df[f'social_{col}'] = social_df[col]
        
        return features_df
    
    def _calculate_social_behavior(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula comportamento social."""
        profile = {}
        
        # Análise de horários para identificar comportamento social
        if 'timestamp' in user_df.columns:
            timestamps = pd.to_datetime(user_df['timestamp'])
            hours = timestamps.dt.hour
            
            # Horários sociais (18-23h) vs solitários (0-6h)
            social_hours = ((hours >= 18) & (hours <= 23)).sum()
            solitary_hours = ((hours >= 0) & (hours <= 6)).sum()
            total_hours = len(hours)
            
            if total_hours > 0:
                profile['social_hours_ratio'] = social_hours / total_hours
                profile['solitary_hours_ratio'] = solitary_hours / total_hours
        
        # Uso de promoções como indicador social
        if 'promotion_used' in user_df.columns:
            promotions = user_df['promotion_used'].fillna(0)
            profile['promotion_engagement'] = promotions.mean()
        
        return profile
    
    def _calculate_single_user_social(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula comportamento social para dados agregados."""
        profile = {}
        
        # Horário preferido como proxy de comportamento social
        if 'preferred_hour' in user_row:
            preferred_hour = user_row.get('preferred_hour', 12)
            
            # Classificar horários
            if 18 <= preferred_hour <= 23:
                profile['social_timing'] = 1  # Horário social
            elif 0 <= preferred_hour <= 6:
                profile['social_timing'] = 0  # Horário solitário
            else:
                profile['social_timing'] = 0.5  # Horário neutro
        
        # Uso de cashback como indicador de engajamento com promoções
        if 'cashback_usage' in user_row:
            cashback = user_row.get('cashback_usage', 0)
            profile['promotion_engagement'] = float(cashback)
        
        return profile
    
    def get_feature_importance_analysis(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa importância e correlações das features extraídas.
        
        Args:
            features_df: DataFrame com features extraídas
            
        Returns:
            Dict com análise de importância
        """
        analysis = {}
        
        # Correlações entre features
        numeric_features = features_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            correlation_matrix = numeric_features.corr()
            
            # Features com alta correlação (potencial redundância)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            analysis['high_correlations'] = high_corr_pairs
        
        # Estatísticas descritivas
        analysis['feature_statistics'] = numeric_features.describe().to_dict()
        
        # Features com baixa variância (potencial remoção)
        low_variance_features = []
        for col in numeric_features.columns:
            variance = numeric_features[col].var()
            if variance < 0.01:  # Threshold baixo
                low_variance_features.append(col)
        
        analysis['low_variance_features'] = low_variance_features
        
        # Distribuição de valores nulos
        null_analysis = {}
        for col in features_df.columns:
            null_count = features_df[col].isnull().sum()
            null_ratio = null_count / len(features_df)
            null_analysis[col] = {'count': null_count, 'ratio': null_ratio}
        
        analysis['null_analysis'] = null_analysis
        
        return analysis
    
    def export_feature_documentation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exporta documentação das features criadas.
        
        Args:
            features_df: DataFrame com features
            
        Returns:
            pd.DataFrame: Documentação das features
        """
        documentation = []
        
        feature_descriptions = {
            # Risk features
            'risk_bet_volatility': 'Volatilidade das apostas (CV)',
            'risk_bet_escalation': 'Tendência de escalação das apostas',
            'risk_bet_p95_ratio': 'Ratio percentil 95 vs mediana',
            'risk_impulsivity_score': 'Score de impulsividade nas apostas',
            'risk_loss_chasing_tendency': 'Tendência de chase após perdas',
            
            # Consistency features
            'consistency_session_regularity': 'Regularidade temporal das sessões',
            'consistency_duration_consistency': 'Consistência na duração',
            'consistency_games_consistency': 'Consistência no número de jogos',
            
            # Engagement features
            'engagement_games_per_minute': 'Jogos por minuto (eficiência)',
            'engagement_session_intensity': 'Intensidade média das sessões',
            'engagement_trend': 'Tendência de engajamento ao longo do tempo',
            'engagement_recency_score': 'Score baseado na atividade recente',
            
            # Diversity features
            'diversity_game_entropy': 'Entropia dos tipos de jogos',
            'diversity_game_concentration': 'Concentração em tipos específicos',
            'diversity_switching_rate': 'Taxa de mudança entre jogos',
            
            # Learning features
            'learning_performance_improvement': 'Melhoria no desempenho',
            'learning_strategy_adaptation': 'Adaptação de estratégia',
            'learning_skill_indicator': 'Indicador de skill',
            
            # Social features
            'social_timing': 'Preferência por horários sociais',
            'social_promotion_engagement': 'Engajamento com promoções'
        }
        
        for feature in features_df.columns:
            doc_entry = {
                'feature_name': feature,
                'description': feature_descriptions.get(feature, 'Feature comportamental'),
                'data_type': str(features_df[feature].dtype),
                'non_null_count': features_df[feature].count(),
                'null_ratio': features_df[feature].isnull().sum() / len(features_df),
                'mean': features_df[feature].mean() if np.issubdtype(features_df[feature].dtype, np.number) else None,
                'std': features_df[feature].std() if np.issubdtype(features_df[feature].dtype, np.number) else None,
                'min': features_df[feature].min() if np.issubdtype(features_df[feature].dtype, np.number) else None,
                'max': features_df[feature].max() if np.issubdtype(features_df[feature].dtype, np.number) else None
            }
            documentation.append(doc_entry)
        
        return pd.DataFrame(documentation)


# Exemplo de uso e teste
if __name__ == "__main__":
    # Dados de exemplo para teste
    np.random.seed(42)
    
    # Simular dados de usuários com transações
    n_users = 500
    n_transactions_per_user = np.random.poisson(20, n_users)
    
    test_data = []
    
    for user_id in range(n_users):
        n_trans = n_transactions_per_user[user_id]
        
        # Simular timestamps
        base_time = pd.Timestamp('2024-01-01')
        timestamps = [base_time + pd.Timedelta(days=np.random.exponential(2)) for _ in range(n_trans)]
        timestamps.sort()
        
        for i, timestamp in enumerate(timestamps):
            # Diferentes perfis comportamentais
            if user_id < 100:  # Usuários conservadores
                bet_amount = np.random.lognormal(1, 0.5)
                session_duration = np.random.exponential(20)
                games_played = np.random.poisson(3)
                game_type = np.random.choice(['slots', 'blackjack'], p=[0.7, 0.3])
            elif user_id < 200:  # Usuários arriscados
                bet_amount = np.random.lognormal(3, 1.5)
                session_duration = np.random.exponential(60)
                games_played = np.random.poisson(8)
                game_type = np.random.choice(['crash', 'roulette', 'sports'], p=[0.4, 0.3, 0.3])
            else:  # Usuários diversos
                bet_amount = np.random.lognormal(2, 1)
                session_duration = np.random.exponential(40)
                games_played = np.random.poisson(5)
                game_type = np.random.choice(['slots', 'sports', 'blackjack', 'crash'], p=[0.3, 0.3, 0.2, 0.2])
            
            # Win amount simulado
            win_amount = bet_amount * np.random.lognormal(0, 0.8) if np.random.random() < 0.45 else 0
            
            test_data.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'bet_amount': bet_amount,
                'win_amount': win_amount,
                'session_duration': session_duration,
                'games_played': games_played,
                'game_type': game_type,
                'promotion_used': np.random.binomial(1, 0.3)
            })
    
    test_df = pd.DataFrame(test_data)
    
    # Testar extração de features
    print("🚀 Testando Behavioral Feature Engine...")
    
    engine = BehavioralFeatureEngine()
    behavioral_features = engine.extract_all_features(test_df)
    
    print(f"✅ Features extraídas: {len(behavioral_features.columns)}")
    print(f"✅ Usuários processados: {len(behavioral_features)}")
    
    # Mostrar algumas features
    print(f"\n✅ Primeiras features comportamentais:")
    print(behavioral_features.head())
    
    # Análise de importância
    importance_analysis = engine.get_feature_importance_analysis(behavioral_features)
    
    print(f"\n✅ Features com baixa variância: {len(importance_analysis['low_variance_features'])}")
    print(f"✅ Pares com alta correlação: {len(importance_analysis['high_correlations'])}")
    
    # Documentação
    documentation = engine.export_feature_documentation(behavioral_features)
    print(f"\n✅ Documentação gerada para {len(documentation)} features")
    
    print("\n🎯 Behavioral Feature Engine implementado com sucesso!")