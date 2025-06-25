"""
Communication Features Engineering - Extração de padrões de comunicação e engajamento
Implementação científica para analisar comportamentos de comunicação e resposta a campanhas.

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
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class CommunicationFeatureEngine:
    """
    Engine para extração de features de comunicação e engajamento de usuários gaming/apostas.
    
    Features extraídas:
    - Padrões de canal preferencial
    - Responsividade a campanhas
    - Engajamento com promoções
    - Comportamento de suporte
    - Preferências de comunicação
    - Timing de interações
    """
    
    def __init__(self, 
                 channels: List[str] = None,
                 campaign_types: List[str] = None,
                 response_window_hours: int = 24,
                 engagement_threshold: float = 0.1):
        """
        Inicializa o engine de features de comunicação.
        
        Args:
            channels: Lista de canais de comunicação disponíveis
            campaign_types: Tipos de campanhas disponíveis
            response_window_hours: Janela para considerar resposta a campanha
            engagement_threshold: Threshold mínimo para considerar engajamento
        """
        self.channels = channels or ['email', 'sms', 'push', 'in_app', 'phone', 'social']
        self.campaign_types = campaign_types or ['welcome', 'retention', 'cashback', 'bonus', 'vip', 'reactivation']
        self.response_window_hours = response_window_hours
        self.engagement_threshold = engagement_threshold
        
        # Caches para otimização
        self._feature_cache = {}
        self._computed_features = set()
        
    def extract_all_features(self, df: pd.DataFrame, 
                           user_id_col: str = 'user_id',
                           interaction_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extrai todas as features de comunicação.
        
        Args:
            df: DataFrame com dados base dos usuários
            user_id_col: Nome da coluna de user_id
            interaction_data: DataFrame opcional com dados de interações/campanhas
            
        Returns:
            pd.DataFrame: DataFrame com features de comunicação
        """
        logger.info("Iniciando extração de features de comunicação...")
        
        # Validar dados de entrada
        self._validate_input_data(df, user_id_col)
        
        # DataFrame para features
        user_ids = df[user_id_col].unique() if user_id_col in df.columns else df.index
        features_df = pd.DataFrame(index=user_ids)
        
        # Agrupar por usuário se necessário
        if user_id_col in df.columns:
            user_data = df.groupby(user_id_col)
        else:
            # Assumir que cada linha é um usuário
            user_data = [(i, row) for i, row in df.iterrows()]
        
        # Preparar dados de interação se disponível
        interaction_grouped = None
        if interaction_data is not None and user_id_col in interaction_data.columns:
            interaction_grouped = interaction_data.groupby(user_id_col)
        
        # Extrair features por categoria
        features_df = self._extract_channel_preference_features(user_data, features_df, interaction_grouped)
        features_df = self._extract_campaign_response_features(user_data, features_df, interaction_grouped)
        features_df = self._extract_engagement_features(user_data, features_df, interaction_grouped)
        features_df = self._extract_support_behavior_features(user_data, features_df, interaction_grouped)
        features_df = self._extract_timing_preferences(user_data, features_df, interaction_grouped)
        features_df = self._extract_content_preferences(user_data, features_df, interaction_grouped)
        
        logger.info(f"Extração concluída. {len(features_df.columns)} features de comunicação criadas")
        return features_df
    
    def _validate_input_data(self, df: pd.DataFrame, user_id_col: str) -> None:
        """Valida se os dados de entrada têm as colunas necessárias."""
        if user_id_col not in df.columns and len(df) > 1000:
            logger.warning(f"Coluna de user_id '{user_id_col}' não encontrada. Assumindo dados agregados.")
    
    def _extract_channel_preference_features(self, user_data: Any, features_df: pd.DataFrame,
                                           interaction_data: Any = None) -> pd.DataFrame:
        """Extrai features de preferência de canal."""
        logger.info("Extraindo features de preferência de canal...")
        
        channel_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                # Buscar dados de interação para este usuário
                user_interactions = None
                if interaction_data is not None:
                    try:
                        user_interactions = interaction_data.get_group(user_id)
                    except KeyError:
                        pass
                
                channel_features[user_id] = self._calculate_channel_preferences(user_df, user_interactions)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    channel_features[idx] = self._calculate_single_user_channels(user_row)
        
        # Converter para DataFrame
        channel_df = pd.DataFrame.from_dict(channel_features, orient='index')
        
        # Combinar com features_df
        for col in channel_df.columns:
            features_df[f'channel_{col}'] = channel_df[col]
        
        return features_df
    
    def _calculate_channel_preferences(self, user_df: pd.DataFrame, 
                                     interaction_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calcula preferências de canal para usuário."""
        profile = {}
        
        if interaction_df is not None and 'channel' in interaction_df.columns:
            channels = interaction_df['channel'].dropna()
            
            if len(channels) > 0:
                # Distribuição de uso por canal
                channel_counts = channels.value_counts()
                total_interactions = len(channels)
                
                # Canal preferido
                preferred_channel = channel_counts.index[0]
                profile['preferred_channel_encoded'] = self._encode_channel(preferred_channel)
                profile['preferred_channel_ratio'] = channel_counts.iloc[0] / total_interactions
                
                # Diversidade de canais
                channel_entropy = entropy(channel_counts / total_interactions)
                profile['channel_diversity'] = channel_entropy / np.log(len(self.channels))
                
                # Ratios por canal específico
                for channel in self.channels:
                    channel_ratio = channel_counts.get(channel, 0) / total_interactions
                    profile[f'{channel}_usage_ratio'] = channel_ratio
                
                # Consistência de canal (concentração no preferido)
                profile['channel_consistency'] = 1 - profile['channel_diversity']
        else:
            # Valores padrão se não há dados de interação
            profile = self._get_default_channel_profile()
        
        return profile
    
    def _calculate_single_user_channels(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula preferências de canal para dados agregados."""
        profile = {}
        
        # Buscar colunas relacionadas a canais
        channel_cols = [col for col in user_row.index if any(ch in col.lower() for ch in self.channels)]
        
        if channel_cols:
            # Se há informações de canal específicas
            for channel in self.channels:
                channel_col = f'{channel}_usage'
                if channel_col in user_row.index:
                    profile[f'{channel}_usage_ratio'] = user_row.get(channel_col, 0)
        else:
            # Inferir baseado em outros dados
            profile = self._infer_channel_preferences(user_row)
        
        return profile
    
    def _encode_channel(self, channel: str) -> int:
        """Codifica canal como número inteiro."""
        try:
            return self.channels.index(channel.lower())
        except ValueError:
            return 0  # Default para primeiro canal
    
    def _get_default_channel_profile(self) -> Dict[str, float]:
        """Retorna perfil de canal padrão."""
        profile = {
            'preferred_channel_encoded': 0,
            'preferred_channel_ratio': 0.6,
            'channel_diversity': 0.5,
            'channel_consistency': 0.5
        }
        
        # Distribuição padrão por canal
        for i, channel in enumerate(self.channels):
            if i == 0:  # Primeiro canal é preferido
                profile[f'{channel}_usage_ratio'] = 0.6
            else:
                profile[f'{channel}_usage_ratio'] = 0.4 / (len(self.channels) - 1)
        
        return profile
    
    def _infer_channel_preferences(self, user_row: pd.Series) -> Dict[str, float]:
        """Infere preferências de canal baseado em outros dados."""
        profile = {}
        
        # Heurísticas baseadas em características do usuário
        age_col = None
        for col in ['age', 'user_age', 'age_group']:
            if col in user_row.index:
                age_col = col
                break
        
        if age_col:
            age = user_row.get(age_col, 30)
            
            # Usuários mais jovens preferem push/in_app
            if age < 25:
                profile['push_usage_ratio'] = 0.4
                profile['in_app_usage_ratio'] = 0.3
                profile['email_usage_ratio'] = 0.2
                profile['sms_usage_ratio'] = 0.1
            elif age < 40:
                profile['email_usage_ratio'] = 0.4
                profile['push_usage_ratio'] = 0.3
                profile['sms_usage_ratio'] = 0.2
                profile['in_app_usage_ratio'] = 0.1
            else:
                # Usuários mais velhos preferem email/sms
                profile['email_usage_ratio'] = 0.5
                profile['sms_usage_ratio'] = 0.3
                profile['phone_usage_ratio'] = 0.2
        
        # Preencher canais não definidos
        for channel in self.channels:
            if f'{channel}_usage_ratio' not in profile:
                profile[f'{channel}_usage_ratio'] = 0.1
        
        return profile
    
    def _extract_campaign_response_features(self, user_data: Any, features_df: pd.DataFrame,
                                          interaction_data: Any = None) -> pd.DataFrame:
        """Extrai features de resposta a campanhas."""
        logger.info("Extraindo features de resposta a campanhas...")
        
        campaign_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                user_interactions = None
                if interaction_data is not None:
                    try:
                        user_interactions = interaction_data.get_group(user_id)
                    except KeyError:
                        pass
                
                campaign_features[user_id] = self._calculate_campaign_response(user_df, user_interactions)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    campaign_features[idx] = self._calculate_single_user_campaigns(user_row)
        
        # Converter para DataFrame
        campaign_df = pd.DataFrame.from_dict(campaign_features, orient='index')
        
        # Combinar com features_df
        for col in campaign_df.columns:
            features_df[f'campaign_{col}'] = campaign_df[col]
        
        return features_df
    
    def _calculate_campaign_response(self, user_df: pd.DataFrame, 
                                   interaction_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calcula resposta a campanhas."""
        profile = {}
        
        if interaction_df is not None:
            # Filtrar campanhas
            campaigns = interaction_df[interaction_df['type'] == 'campaign'] if 'type' in interaction_df.columns else interaction_df
            
            if len(campaigns) > 0:
                # Taxa de resposta geral
                if 'responded' in campaigns.columns:
                    responses = campaigns['responded'].fillna(0)
                    profile['overall_response_rate'] = responses.mean()
                    profile['total_campaigns_received'] = len(campaigns)
                    profile['total_responses'] = responses.sum()
                
                # Análise por tipo de campanha
                if 'campaign_type' in campaigns.columns:
                    campaign_types = campaigns['campaign_type'].dropna()
                    
                    for camp_type in self.campaign_types:
                        type_campaigns = campaigns[campaigns['campaign_type'] == camp_type]
                        
                        if len(type_campaigns) > 0:
                            if 'responded' in type_campaigns.columns:
                                type_response_rate = type_campaigns['responded'].fillna(0).mean()
                                profile[f'{camp_type}_response_rate'] = type_response_rate
                
                # Timing de resposta
                if 'timestamp' in campaigns.columns and 'response_timestamp' in campaigns.columns:
                    campaign_times = pd.to_datetime(campaigns['timestamp'])
                    response_times = pd.to_datetime(campaigns['response_timestamp'])
                    
                    # Calcular tempo de resposta
                    response_delays = (response_times - campaign_times).dt.total_seconds() / 3600  # em horas
                    response_delays_clean = response_delays.dropna()
                    
                    if len(response_delays_clean) > 0:
                        profile['avg_response_time_hours'] = response_delays_clean.mean()
                        profile['fast_response_ratio'] = (response_delays_clean <= self.response_window_hours).mean()
        
        return profile
    
    def _calculate_single_user_campaigns(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula resposta a campanhas para dados agregados."""
        profile = {}
        
        # Buscar métricas de campanha agregadas
        campaign_cols = [col for col in user_row.index if 'campaign' in col.lower() or 'response' in col.lower()]
        
        if 'campaign_response_rate' in user_row.index:
            profile['overall_response_rate'] = user_row.get('campaign_response_rate', 0.1)
        
        # Response rates por tipo se disponível
        for camp_type in self.campaign_types:
            col_name = f'{camp_type}_response_rate'
            if col_name in user_row.index:
                profile[col_name] = user_row.get(col_name, 0.1)
        
        return profile
    
    def _extract_engagement_features(self, user_data: Any, features_df: pd.DataFrame,
                                   interaction_data: Any = None) -> pd.DataFrame:
        """Extrai features de engajamento geral."""
        logger.info("Extraindo features de engajamento...")
        
        engagement_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                user_interactions = None
                if interaction_data is not None:
                    try:
                        user_interactions = interaction_data.get_group(user_id)
                    except KeyError:
                        pass
                
                engagement_features[user_id] = self._calculate_engagement_level(user_df, user_interactions)
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
    
    def _calculate_engagement_level(self, user_df: pd.DataFrame, 
                                  interaction_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calcula nível de engajamento."""
        profile = {}
        
        if interaction_df is not None:
            # Métricas básicas de engajamento
            total_interactions = len(interaction_df)
            profile['total_interactions'] = total_interactions
            
            # Frequência de interação
            if 'timestamp' in interaction_df.columns:
                timestamps = pd.to_datetime(interaction_df['timestamp'])
                time_span = (timestamps.max() - timestamps.min()).total_seconds() / 86400  # dias
                
                if time_span > 0:
                    profile['interaction_frequency'] = total_interactions / time_span
            
            # Tipos de interação
            if 'interaction_type' in interaction_df.columns:
                interaction_types = interaction_df['interaction_type'].value_counts()
                
                # Diversidade de tipos de interação
                type_entropy = entropy(interaction_types / total_interactions)
                profile['interaction_diversity'] = type_entropy
                
                # Interações proativas vs reativas
                proactive_types = ['support_initiated', 'feedback', 'complaint']
                reactive_types = ['campaign_response', 'promotion_click']
                
                proactive_count = sum(interaction_types.get(ptype, 0) for ptype in proactive_types)
                reactive_count = sum(interaction_types.get(rtype, 0) for rtype in reactive_types)
                
                if total_interactions > 0:
                    profile['proactive_interaction_ratio'] = proactive_count / total_interactions
                    profile['reactive_interaction_ratio'] = reactive_count / total_interactions
            
            # Engajamento com promoções
            if 'promotion_used' in interaction_df.columns:
                promotion_usage = interaction_df['promotion_used'].fillna(0)
                profile['promotion_engagement'] = promotion_usage.mean()
        
        return profile
    
    def _calculate_single_user_engagement(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula engajamento para dados agregados."""
        profile = {}
        
        # Buscar indicadores de engajamento
        engagement_indicators = ['promotion_usage', 'cashback_usage', 'support_tickets', 'app_opens']
        
        engagement_score = 0
        indicator_count = 0
        
        for indicator in engagement_indicators:
            if indicator in user_row.index:
                value = user_row.get(indicator, 0)
                # Normalizar valor (assumindo máximo razoável)
                if 'usage' in indicator:
                    normalized_value = min(value, 1)  # Já é ratio
                else:
                    normalized_value = min(value / 10, 1)  # Normalizar por 10
                
                engagement_score += normalized_value
                indicator_count += 1
        
        if indicator_count > 0:
            profile['engagement_score'] = engagement_score / indicator_count
        
        return profile
    
    def _extract_support_behavior_features(self, user_data: Any, features_df: pd.DataFrame,
                                         interaction_data: Any = None) -> pd.DataFrame:
        """Extrai features de comportamento de suporte."""
        logger.info("Extraindo features de comportamento de suporte...")
        
        support_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                user_interactions = None
                if interaction_data is not None:
                    try:
                        user_interactions = interaction_data.get_group(user_id)
                    except KeyError:
                        pass
                
                support_features[user_id] = self._calculate_support_behavior(user_df, user_interactions)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    support_features[idx] = self._calculate_single_user_support(user_row)
        
        # Converter para DataFrame
        support_df = pd.DataFrame.from_dict(support_features, orient='index')
        
        # Combinar com features_df
        for col in support_df.columns:
            features_df[f'support_{col}'] = support_df[col]
        
        return features_df
    
    def _calculate_support_behavior(self, user_df: pd.DataFrame, 
                                  interaction_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calcula comportamento de suporte."""
        profile = {}
        
        if interaction_df is not None:
            # Filtrar interações de suporte
            support_interactions = interaction_df[
                interaction_df['interaction_type'].str.contains('support', case=False, na=False)
            ] if 'interaction_type' in interaction_df.columns else pd.DataFrame()
            
            if len(support_interactions) > 0:
                profile['support_tickets_count'] = len(support_interactions)
                
                # Frequência de tickets
                if 'timestamp' in support_interactions.columns:
                    timestamps = pd.to_datetime(support_interactions['timestamp'])
                    time_span = (timestamps.max() - timestamps.min()).total_seconds() / 86400
                    
                    if time_span > 0:
                        profile['support_frequency'] = len(support_interactions) / time_span
                
                # Categorias de suporte
                if 'category' in support_interactions.columns:
                    categories = support_interactions['category'].value_counts()
                    
                    # Categoria mais comum
                    if len(categories) > 0:
                        profile['most_common_issue'] = self._encode_support_category(categories.index[0])
                        profile['issue_diversity'] = entropy(categories / len(support_interactions))
                
                # Satisfação com suporte
                if 'satisfaction' in support_interactions.columns:
                    satisfaction = support_interactions['satisfaction'].dropna()
                    if len(satisfaction) > 0:
                        profile['avg_support_satisfaction'] = satisfaction.mean()
        
        return profile
    
    def _calculate_single_user_support(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula suporte para dados agregados."""
        profile = {}
        
        # Buscar métricas de suporte
        if 'support_tickets' in user_row.index:
            profile['support_tickets_count'] = user_row.get('support_tickets', 0)
        
        if 'support_satisfaction' in user_row.index:
            profile['avg_support_satisfaction'] = user_row.get('support_satisfaction', 3.5)
        
        return profile
    
    def _encode_support_category(self, category: str) -> int:
        """Codifica categoria de suporte como número."""
        categories = ['payment', 'technical', 'account', 'bonus', 'general']
        try:
            return categories.index(category.lower())
        except ValueError:
            return 4  # 'general' como padrão
    
    def _extract_timing_preferences(self, user_data: Any, features_df: pd.DataFrame,
                                  interaction_data: Any = None) -> pd.DataFrame:
        """Extrai preferências de timing para comunicação."""
        logger.info("Extraindo preferências de timing...")
        
        timing_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                user_interactions = None
                if interaction_data is not None:
                    try:
                        user_interactions = interaction_data.get_group(user_id)
                    except KeyError:
                        pass
                
                timing_features[user_id] = self._calculate_timing_preferences(user_df, user_interactions)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    timing_features[idx] = self._calculate_single_user_timing(user_row)
        
        # Converter para DataFrame
        timing_df = pd.DataFrame.from_dict(timing_features, orient='index')
        
        # Combinar com features_df
        for col in timing_df.columns:
            features_df[f'timing_{col}'] = timing_df[col]
        
        return features_df
    
    def _calculate_timing_preferences(self, user_df: pd.DataFrame, 
                                    interaction_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calcula preferências de timing."""
        profile = {}
        
        if interaction_df is not None and 'timestamp' in interaction_df.columns:
            timestamps = pd.to_datetime(interaction_df['timestamp'])
            
            # Análise de horários
            hours = timestamps.dt.hour
            
            if len(hours) > 0:
                profile['preferred_contact_hour'] = hours.mode().iloc[0] if not hours.mode().empty else hours.mean()
                
                # Distribuição por período do dia
                morning = ((hours >= 6) & (hours < 12)).mean()
                afternoon = ((hours >= 12) & (hours < 18)).mean()
                evening = ((hours >= 18) & (hours < 22)).mean()
                night = ((hours >= 22) | (hours < 6)).mean()
                
                profile['morning_interaction_ratio'] = morning
                profile['afternoon_interaction_ratio'] = afternoon
                profile['evening_interaction_ratio'] = evening
                profile['night_interaction_ratio'] = night
            
            # Análise de dias da semana
            weekdays = timestamps.dt.dayofweek
            weekend_ratio = weekdays.isin([5, 6]).mean()
            profile['weekend_interaction_ratio'] = weekend_ratio
        
        return profile
    
    def _calculate_single_user_timing(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula timing para dados agregados."""
        profile = {}
        
        # Usar horário preferido de jogo como proxy
        if 'preferred_hour' in user_row.index:
            preferred_hour = user_row.get('preferred_hour', 20)
            profile['preferred_contact_hour'] = preferred_hour
            
            # Inferir período do dia
            if 6 <= preferred_hour < 12:
                profile['morning_interaction_ratio'] = 0.8
                profile['afternoon_interaction_ratio'] = 0.2
            elif 12 <= preferred_hour < 18:
                profile['afternoon_interaction_ratio'] = 0.8
                profile['evening_interaction_ratio'] = 0.2
            elif 18 <= preferred_hour < 22:
                profile['evening_interaction_ratio'] = 0.8
                profile['afternoon_interaction_ratio'] = 0.2
            else:
                profile['night_interaction_ratio'] = 0.8
                profile['evening_interaction_ratio'] = 0.2
        
        # Preencher valores ausentes
        for period in ['morning', 'afternoon', 'evening', 'night']:
            key = f'{period}_interaction_ratio'
            if key not in profile:
                profile[key] = 0.25  # Distribuição uniforme padrão
        
        return profile
    
    def _extract_content_preferences(self, user_data: Any, features_df: pd.DataFrame,
                                   interaction_data: Any = None) -> pd.DataFrame:
        """Extrai preferências de conteúdo."""
        logger.info("Extraindo preferências de conteúdo...")
        
        content_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                user_interactions = None
                if interaction_data is not None:
                    try:
                        user_interactions = interaction_data.get_group(user_id)
                    except KeyError:
                        pass
                
                content_features[user_id] = self._calculate_content_preferences(user_df, user_interactions)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    content_features[idx] = self._calculate_single_user_content(user_row)
        
        # Converter para DataFrame
        content_df = pd.DataFrame.from_dict(content_features, orient='index')
        
        # Combinar com features_df
        for col in content_df.columns:
            features_df[f'content_{col}'] = content_df[col]
        
        return features_df
    
    def _calculate_content_preferences(self, user_df: pd.DataFrame, 
                                     interaction_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calcula preferências de conteúdo."""
        profile = {}
        
        if interaction_df is not None:
            # Analisar tipos de conteúdo que geram mais engajamento
            if 'content_type' in interaction_df.columns and 'engagement_score' in interaction_df.columns:
                content_engagement = interaction_df.groupby('content_type')['engagement_score'].mean()
                
                if len(content_engagement) > 0:
                    # Tipo de conteúdo preferido
                    preferred_content = content_engagement.idxmax()
                    profile['preferred_content_type'] = self._encode_content_type(preferred_content)
                    profile['content_preference_strength'] = content_engagement.max()
            
            # Análise de promoções
            promotion_interactions = interaction_df[
                interaction_df['content_type'].str.contains('promo', case=False, na=False)
            ] if 'content_type' in interaction_df.columns else pd.DataFrame()
            
            if len(promotion_interactions) > 0:
                total_interactions = len(interaction_df)
                profile['promotion_content_ratio'] = len(promotion_interactions) / total_interactions
        
        return profile
    
    def _calculate_single_user_content(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula conteúdo para dados agregados."""
        profile = {}
        
        # Inferir preferências baseado em uso de promoções
        if 'cashback_usage' in user_row.index:
            cashback_usage = user_row.get('cashback_usage', 0)
            profile['promotion_content_ratio'] = float(cashback_usage)
        
        return profile
    
    def _encode_content_type(self, content_type: str) -> int:
        """Codifica tipo de conteúdo como número."""
        content_types = ['promotional', 'educational', 'entertainment', 'transactional', 'social']
        try:
            return content_types.index(content_type.lower())
        except ValueError:
            return 0  # 'promotional' como padrão
    
    def get_communication_insights(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights das features de comunicação extraídas.
        
        Args:
            features_df: DataFrame com features de comunicação
            
        Returns:
            Dict com insights de comunicação
        """
        insights = {}
        
        # Análise de canais preferenciais
        channel_cols = [col for col in features_df.columns if 'channel_' in col and '_usage_ratio' in col]
        if channel_cols:
            channel_preferences = {}
            for col in channel_cols:
                channel_name = col.replace('channel_', '').replace('_usage_ratio', '')
                channel_preferences[channel_name] = features_df[col].mean()
            
            insights['channel_preferences'] = dict(sorted(channel_preferences.items(), 
                                                         key=lambda x: x[1], reverse=True))
        
        # Análise de responsividade
        if 'campaign_overall_response_rate' in features_df.columns:
            response_rates = features_df['campaign_overall_response_rate'].dropna()
            
            insights['campaign_responsiveness'] = {
                'avg_response_rate': response_rates.mean(),
                'highly_responsive_users': (response_rates > 0.3).mean(),
                'low_responsive_users': (response_rates < 0.1).mean()
            }
        
        # Análise de timing
        if 'timing_preferred_contact_hour' in features_df.columns:
            preferred_hours = features_df['timing_preferred_contact_hour'].dropna()
            
            insights['optimal_contact_timing'] = {
                'peak_hours': preferred_hours.mode().tolist(),
                'avg_preferred_hour': preferred_hours.mean(),
                'hour_distribution': preferred_hours.value_counts().head(5).to_dict()
            }
        
        return insights
    
    def export_communication_documentation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exporta documentação das features de comunicação.
        
        Args:
            features_df: DataFrame com features
            
        Returns:
            pd.DataFrame: Documentação das features
        """
        documentation = []
        
        feature_descriptions = {
            # Channel features
            'channel_preferred_channel_encoded': 'Canal preferido codificado (0=email, 1=sms, etc)',
            'channel_channel_diversity': 'Diversidade de uso de canais (0-1)',
            'channel_email_usage_ratio': 'Proporção de uso do canal email',
            'channel_sms_usage_ratio': 'Proporção de uso do canal SMS',
            'channel_push_usage_ratio': 'Proporção de uso de push notifications',
            
            # Campaign features
            'campaign_overall_response_rate': 'Taxa geral de resposta a campanhas',
            'campaign_welcome_response_rate': 'Taxa de resposta a campanhas de boas-vindas',
            'campaign_retention_response_rate': 'Taxa de resposta a campanhas de retenção',
            'campaign_avg_response_time_hours': 'Tempo médio de resposta em horas',
            
            # Engagement features
            'engagement_interaction_frequency': 'Frequência de interações por dia',
            'engagement_promotion_engagement': 'Engajamento com promoções (0-1)',
            'engagement_proactive_interaction_ratio': 'Proporção de interações proativas',
            
            # Support features
            'support_tickets_count': 'Número total de tickets de suporte',
            'support_avg_support_satisfaction': 'Satisfação média com suporte (1-5)',
            
            # Timing features
            'timing_preferred_contact_hour': 'Horário preferido para contato (0-23)',
            'timing_evening_interaction_ratio': 'Proporção de interações à noite',
            'timing_weekend_interaction_ratio': 'Proporção de interações em fins de semana',
            
            # Content features
            'content_promotion_content_ratio': 'Proporção de engajamento com conteúdo promocional'
        }
        
        for feature in features_df.columns:
            doc_entry = {
                'feature_name': feature,
                'category': feature.split('_')[0] if '_' in feature else 'communication',
                'description': feature_descriptions.get(feature, 'Feature de comunicação'),
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
    
    # Simular dados de usuários
    n_users = 200
    
    user_data = []
    interaction_data = []
    
    for user_id in range(n_users):
        # Dados básicos do usuário
        age = np.random.randint(18, 65)
        user_data.append({
            'user_id': user_id,
            'age': age,
            'cashback_usage': np.random.binomial(1, 0.4),
            'preferred_hour': np.random.randint(0, 24)
        })
        
        # Simular interações
        n_interactions = np.random.poisson(15) + 1
        
        for interaction_id in range(n_interactions):
            # Diferentes tipos de usuário têm diferentes padrões
            if user_id < 50:  # Usuários altamente engajados
                channels = ['email', 'push', 'in_app']
                response_prob = 0.4
            elif user_id < 100:  # Usuários moderadamente engajados
                channels = ['email', 'sms']
                response_prob = 0.2
            else:  # Usuários baixo engajamento
                channels = ['email']
                response_prob = 0.1
            
            channel = np.random.choice(channels)
            interaction_type = np.random.choice(['campaign', 'support', 'promotion', 'notification'])
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, 90))
            
            interaction_data.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'channel': channel,
                'type': interaction_type,
                'interaction_type': f'{interaction_type}_{channel}',
                'responded': np.random.binomial(1, response_prob),
                'campaign_type': np.random.choice(['welcome', 'retention', 'cashback', 'bonus']),
                'content_type': np.random.choice(['promotional', 'educational', 'transactional']),
                'engagement_score': np.random.uniform(0, 1)
            })
    
    user_df = pd.DataFrame(user_data)
    interaction_df = pd.DataFrame(interaction_data)
    
    # Testar extração de features de comunicação
    print("🚀 Testando Communication Feature Engine...")
    
    engine = CommunicationFeatureEngine()
    communication_features = engine.extract_all_features(user_df, interaction_data=interaction_df)
    
    print(f"✅ Features de comunicação extraídas: {len(communication_features.columns)}")
    print(f"✅ Usuários processados: {len(communication_features)}")
    
    # Mostrar algumas features
    print(f"\n✅ Primeiras features de comunicação:")
    print(communication_features.head())
    
    # Insights de comunicação
    insights = engine.get_communication_insights(communication_features)
    
    if 'channel_preferences' in insights:
        print(f"\n✅ Canais preferenciais:")
        for channel, usage in list(insights['channel_preferences'].items())[:3]:
            print(f"   {channel}: {usage:.1%}")
    
    if 'campaign_responsiveness' in insights:
        responsiveness = insights['campaign_responsiveness']
        print(f"\n✅ Responsividade a campanhas:")
        print(f"   Taxa média de resposta: {responsiveness['avg_response_rate']:.1%}")
        print(f"   Usuários altamente responsivos: {responsiveness['highly_responsive_users']:.1%}")
    
    # Documentação
    documentation = engine.export_communication_documentation(communication_features)
    print(f"\n✅ Documentação gerada para {len(documentation)} features de comunicação")
    
    print("\n🎯 Communication Feature Engine implementado com sucesso!")