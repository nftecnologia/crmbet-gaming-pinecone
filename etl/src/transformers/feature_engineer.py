"""
🚀 Feature Engineer - Criação de Features para ML
Engenharia de features inteligente para segmentação de usuários

Author: Agente Engenheiro de Dados - ULTRATHINK
Created: 2025-06-25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import structlog
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from collections import Counter
import json

logger = structlog.get_logger(__name__)

@dataclass
class FeatureConfig:
    """Configuração de feature engineering"""
    
    # Features de Comportamento
    include_behavioral_features: bool = True
    include_temporal_features: bool = True
    include_transaction_features: bool = True
    include_gaming_features: bool = True
    
    # Períodos de Análise
    analysis_periods: Dict[str, int] = field(default_factory=lambda: {
        'last_7_days': 7,
        'last_30_days': 30,
        'last_90_days': 90
    })
    
    # Categorias de Ticket Médio
    ticket_categories: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'baixo': (0.0, 50.0),
        'medio': (50.0, 500.0),
        'alto': (500.0, float('inf'))
    })
    
    # Gaming Features
    game_types: List[str] = field(default_factory=lambda: [
        'crash', 'cassino', 'esportes', 'poker', 'slots', 'roleta', 'blackjack'
    ])
    
    # Canais de Comunicação
    communication_channels: List[str] = field(default_factory=lambda: [
        'email', 'sms', 'push', 'whatsapp', 'telegram', 'phone'
    ])
    
    # Horários de Atividade (slots de 4h)
    activity_time_slots: List[str] = field(default_factory=lambda: [
        'madrugada', 'manha', 'tarde', 'noite'  # 0-6, 6-12, 12-18, 18-24
    ])
    
    # Thresholds
    min_transactions_for_pattern: int = 5
    outlier_percentile: float = 95.0
    
    # Performance
    enable_parallel_processing: bool = True
    chunk_size: int = 10000

@dataclass
class FeatureReport:
    """Relatório de feature engineering"""
    original_features: int = 0
    created_features: int = 0
    feature_categories: Dict[str, int] = field(default_factory=dict)
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    
    @property
    def total_features(self) -> int:
        return self.original_features + self.created_features

class FeatureEngineer:
    """
    Engenheiro de Features para ML de Segmentação
    Foco em padrões comportamentais de usuários de apostas/gaming
    """
    
    def __init__(self, target_features: Optional[List[str]] = None, 
                 config: Optional[FeatureConfig] = None):
        
        self.config = config or FeatureConfig()
        self.target_features = target_features or [
            'favorite_game_type',
            'ticket_medio_categoria', 
            'dias_semana_preferidos',
            'horarios_atividade',
            'canal_comunicacao_preferido',
            'frequencia_jogo',
            'padrao_deposito',
            'padrao_saque'
        ]
        
        self.logger = logger.bind(component="FeatureEngineer")
        
        # Cache de cálculos
        self._feature_cache = {}
        
        # Encoders
        self._label_encoders = {}
        self._scaler = StandardScaler()
        
        self.logger.info(
            "FeatureEngineer inicializado",
            target_features=len(self.target_features),
            config=self.config.__dict__
        )
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline principal de feature engineering
        
        Args:
            df: DataFrame limpo com dados de usuários
            
        Returns:
            DataFrame com features engineered para ML
        """
        if df.empty:
            self.logger.warning("DataFrame vazio fornecido para feature engineering")
            return df
        
        start_time = datetime.now()
        self.logger.info("Iniciando feature engineering", records=len(df))
        
        # Inicializa relatório
        report = FeatureReport(original_features=len(df.columns))
        
        try:
            # Preparação dos dados
            df = self._prepare_base_data(df)
            
            # Fase 1: Features de Gaming/Jogo
            if self.config.include_gaming_features:
                df = self._create_gaming_features(df, report)
            
            # Fase 2: Features Comportamentais
            if self.config.include_behavioral_features:
                df = self._create_behavioral_features(df, report)
            
            # Fase 3: Features Temporais
            if self.config.include_temporal_features:
                df = self._create_temporal_features(df, report)
            
            # Fase 4: Features Transacionais
            if self.config.include_transaction_features:
                df = self._create_transaction_features(df, report)
            
            # Fase 5: Features Agregadas por Usuário
            df = self._create_user_aggregated_features(df, report)
            
            # Fase 6: Features de RFM (Recency, Frequency, Monetary)
            df = self._create_rfm_features(df, report)
            
            # Fase 7: Features de Segmentação
            df = self._create_segmentation_features(df, report)
            
            # Fase 8: Normalização e Encoding
            df = self._normalize_and_encode_features(df, report)
            
            # Finaliza relatório
            report.created_features = len(df.columns) - report.original_features
            report.processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                "Feature engineering concluído",
                original_features=report.original_features,
                created_features=report.created_features,
                total_features=report.total_features,
                processing_time=report.processing_time
            )
            
            return df
            
        except Exception as e:
            self.logger.error("Erro no feature engineering", error=str(e))
            raise
    
    def _prepare_base_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preparação básica dos dados"""
        
        # Garante que temos colunas essenciais
        if 'user_id' not in df.columns:
            raise ValueError("Coluna 'user_id' é obrigatória")
        
        # Converte datas
        date_columns = ['created_at', 'updated_at', 'last_login', 'registration_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ordena por usuário e data
        if 'created_at' in df.columns:
            df = df.sort_values(['user_id', 'created_at'])
        
        return df
    
    def _create_gaming_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features específicas de gaming"""
        
        self.logger.info("Criando features de gaming")
        
        # 1. Jogo Favorito por Usuário
        if 'game_type' in df.columns:
            user_game_preference = df.groupby('user_id')['game_type'].agg([
                lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',  # Mais frequente
                'count',
                'nunique'
            ]).rename(columns={
                '<lambda>': 'favorite_game_type',
                'count': 'total_games_played',
                'nunique': 'unique_games_played'
            })
            
            df = df.merge(user_game_preference, on='user_id', how='left')
        
        # 2. Diversidade de Jogos
        if 'game_type' in df.columns:
            game_diversity = df.groupby('user_id')['game_type'].apply(
                lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
            ).rename('game_diversity_score')
            
            df = df.merge(game_diversity.to_frame(), on='user_id', how='left')
        
        # 3. Padrão de Apostas por Tipo de Jogo
        if 'game_type' in df.columns and 'amount' in df.columns:
            game_bet_patterns = df.groupby(['user_id', 'game_type'])['amount'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            # Pivot para ter padrões de aposta por tipo de jogo
            game_bet_pivot = game_bet_patterns.pivot(
                index='user_id', 
                columns='game_type', 
                values='mean'
            ).fillna(0)
            
            # Renomeia colunas
            game_bet_pivot.columns = [f'avg_bet_{col}' for col in game_bet_pivot.columns]
            
            df = df.merge(game_bet_pivot, on='user_id', how='left')
        
        # 4. Frequência de Jogo por Período
        if 'created_at' in df.columns:
            for period_name, days in self.config.analysis_periods.items():
                cutoff_date = datetime.now() - timedelta(days=days)
                
                recent_games = df[df['created_at'] >= cutoff_date]
                game_frequency = recent_games.groupby('user_id').size().rename(f'games_frequency_{period_name}')
                
                df = df.merge(game_frequency.to_frame(), on='user_id', how='left')
        
        report.feature_categories['gaming'] = len([col for col in df.columns if any(
            keyword in col.lower() for keyword in ['game', 'bet', 'frequency']
        )])
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features comportamentais"""
        
        self.logger.info("Criando features comportamentais")
        
        # 1. Canal de Comunicação Preferido
        if 'communication_channel' in df.columns:
            preferred_channels = df.groupby('user_id')['communication_channel'].agg([
                lambda x: x.mode().iloc[0] if not x.mode().empty else 'email',
                'count'
            ]).rename(columns={
                '<lambda>': 'canal_comunicacao_preferido',
                'count': 'total_communications'
            })
            
            df = df.merge(preferred_channels, on='user_id', how='left')
        
        # 2. Padrão de Atividade por Dispositivo
        if 'device_type' in df.columns:
            device_usage = df.groupby('user_id')['device_type'].agg([
                lambda x: x.mode().iloc[0] if not x.mode().empty else 'desktop',
                'nunique'
            ]).rename(columns={
                '<lambda>': 'preferred_device',
                'nunique': 'device_variety'
            })
            
            df = df.merge(device_usage, on='user_id', how='left')
        
        # 3. Padrão de Sessão
        if 'session_duration' in df.columns:
            session_patterns = df.groupby('user_id')['session_duration'].agg([
                'mean', 'median', 'std', 'max', 'count'
            ]).rename(columns={
                'mean': 'avg_session_duration',
                'median': 'median_session_duration',
                'std': 'session_duration_variability',
                'max': 'max_session_duration',
                'count': 'total_sessions'
            })
            
            df = df.merge(session_patterns, on='user_id', how='left')
        
        # 4. Engagement Score
        engagement_features = []
        if 'total_sessions' in df.columns:
            engagement_features.append('total_sessions')
        if 'total_games_played' in df.columns:
            engagement_features.append('total_games_played')
        if 'avg_session_duration' in df.columns:
            engagement_features.append('avg_session_duration')
        
        if engagement_features:
            # Normaliza e calcula score de engagement
            engagement_data = df[['user_id'] + engagement_features].drop_duplicates('user_id')
            
            scaler = StandardScaler()
            engagement_normalized = scaler.fit_transform(
                engagement_data[engagement_features].fillna(0)
            )
            
            engagement_data['engagement_score'] = np.mean(engagement_normalized, axis=1)
            
            df = df.merge(
                engagement_data[['user_id', 'engagement_score']], 
                on='user_id', 
                how='left'
            )
        
        report.feature_categories['behavioral'] = len([col for col in df.columns if any(
            keyword in col.lower() for keyword in ['canal', 'device', 'session', 'engagement']
        )])
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features temporais"""
        
        self.logger.info("Criando features temporais")
        
        if 'created_at' not in df.columns:
            return df
        
        # 1. Horários de Maior Atividade
        df['hour_of_day'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Categoriza horários em slots
        def categorize_hour(hour):
            if 0 <= hour < 6:
                return 'madrugada'
            elif 6 <= hour < 12:
                return 'manha'
            elif 12 <= hour < 18:
                return 'tarde'
            else:
                return 'noite'
        
        df['time_slot'] = df['hour_of_day'].apply(categorize_hour)
        
        # 2. Padrões Temporais por Usuário
        temporal_patterns = df.groupby('user_id').agg({
            'hour_of_day': lambda x: x.mode().iloc[0] if not x.mode().empty else 12,
            'day_of_week': lambda x: x.mode().iloc[0] if not x.mode().empty else 1,
            'time_slot': lambda x: x.mode().iloc[0] if not x.mode().empty else 'tarde',
            'is_weekend': 'mean'
        }).rename(columns={
            'hour_of_day': 'preferred_hour',
            'day_of_week': 'preferred_day_of_week',
            'time_slot': 'horarios_atividade',
            'is_weekend': 'weekend_activity_ratio'
        })
        
        df = df.merge(temporal_patterns, on='user_id', how='left')
        
        # 3. Dias da Semana Preferidos (Agregação por usuário)
        user_day_preferences = df.groupby('user_id')['day_of_week'].apply(
            lambda x: ','.join(map(str, sorted(x.mode().tolist())))
        ).rename('dias_semana_preferidos')
        
        df = df.merge(user_day_preferences.to_frame(), on='user_id', how='left')
        
        # 4. Regularidade de Atividade
        activity_regularity = df.groupby('user_id')['created_at'].apply(
            lambda x: x.diff().dt.days.std() if len(x) > 1 else 0
        ).rename('activity_regularity')
        
        df = df.merge(activity_regularity.to_frame(), on='user_id', how='left')
        
        # 5. Sazonalidade
        df['month'] = df['created_at'].dt.month
        df['quarter'] = df['created_at'].dt.quarter
        df['day_of_month'] = df['created_at'].dt.day
        
        report.feature_categories['temporal'] = len([col for col in df.columns if any(
            keyword in col.lower() for keyword in ['hour', 'day', 'time', 'weekend', 'month', 'quarter']
        )])
        
        return df
    
    def _create_transaction_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features transacionais"""
        
        self.logger.info("Criando features transacionais")
        
        if 'amount' not in df.columns:
            return df
        
        # 1. Ticket Médio e Categorização
        user_transaction_stats = df.groupby('user_id')['amount'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count', 'sum'
        ]).rename(columns={
            'mean': 'ticket_medio',
            'median': 'ticket_mediano',
            'std': 'ticket_variabilidade',
            'min': 'menor_transacao',
            'max': 'maior_transacao',
            'count': 'total_transacoes',
            'sum': 'volume_total'
        })
        
        # Categoriza ticket médio
        def categorize_ticket(ticket):
            for category, (min_val, max_val) in self.config.ticket_categories.items():
                if min_val <= ticket < max_val:
                    return category
            return 'alto'  # Fallback
        
        user_transaction_stats['ticket_medio_categoria'] = user_transaction_stats['ticket_medio'].apply(categorize_ticket)
        
        df = df.merge(user_transaction_stats, on='user_id', how='left')
        
        # 2. Padrões de Depósito vs Saque
        if 'transaction_type' in df.columns:
            transaction_patterns = df.groupby(['user_id', 'transaction_type'])['amount'].agg([
                'sum', 'count', 'mean'
            ]).reset_index()
            
            # Separa depósitos e saques
            deposits = transaction_patterns[transaction_patterns['transaction_type'].isin(['deposit', 'bonus'])]
            withdrawals = transaction_patterns[transaction_patterns['transaction_type'].isin(['withdrawal', 'win'])]
            
            if not deposits.empty:
                deposit_stats = deposits.groupby('user_id')['sum'].sum().rename('total_deposits')
                deposit_freq = deposits.groupby('user_id')['count'].sum().rename('deposit_frequency')
                
                df = df.merge(deposit_stats.to_frame(), on='user_id', how='left')
                df = df.merge(deposit_freq.to_frame(), on='user_id', how='left')
            
            if not withdrawals.empty:
                withdrawal_stats = withdrawals.groupby('user_id')['sum'].sum().rename('total_withdrawals')
                withdrawal_freq = withdrawals.groupby('user_id')['count'].sum().rename('withdrawal_frequency')
                
                df = df.merge(withdrawal_stats.to_frame(), on='user_id', how='left')
                df = df.merge(withdrawal_freq.to_frame(), on='user_id', how='left')
            
            # Calcula balanço e ratios
            if 'total_deposits' in df.columns and 'total_withdrawals' in df.columns:
                df['balance_ratio'] = df['total_deposits'] / (df['total_withdrawals'] + 1)
                df['net_balance'] = df['total_deposits'].fillna(0) - df['total_withdrawals'].fillna(0)
        
        # 3. Padrões de Frequência de Transação
        transaction_frequency = df.groupby('user_id')['created_at'].apply(
            lambda x: len(x) / ((x.max() - x.min()).days + 1) if len(x) > 1 else 0
        ).rename('frequencia_jogo')
        
        df = df.merge(transaction_frequency.to_frame(), on='user_id', how='left')
        
        # 4. Volatilidade de Apostas
        bet_volatility = df.groupby('user_id')['amount'].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        ).rename('bet_volatility')
        
        df = df.merge(bet_volatility.to_frame(), on='user_id', how='left')
        
        report.feature_categories['transaction'] = len([col for col in df.columns if any(
            keyword in col.lower() for keyword in ['ticket', 'transacao', 'deposit', 'withdrawal', 'balance', 'frequencia']
        )])
        
        return df
    
    def _create_user_aggregated_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features agregadas por usuário"""
        
        self.logger.info("Criando features agregadas por usuário")
        
        # Agrupa todas as features numéricas por usuário
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        user_columns = ['user_id']
        
        # Features já agregadas (não reagregar)
        already_aggregated = [col for col in numeric_columns if any(
            keyword in col.lower() for keyword in ['total_', 'avg_', 'mean_', 'median_', 'max_', 'min_', 'frequency']
        )]
        
        # Colunas para agregação
        columns_to_aggregate = [col for col in numeric_columns if col not in already_aggregated + user_columns]
        
        if columns_to_aggregate:
            user_aggregations = df.groupby('user_id')[columns_to_aggregate].agg([
                'mean', 'std', 'min', 'max'
            ]).round(4)
            
            # Flatten column names
            user_aggregations.columns = [f"{col[0]}_{col[1]}" for col in user_aggregations.columns]
            
            df = df.merge(user_aggregations, on='user_id', how='left')
        
        return df
    
    def _create_rfm_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features RFM (Recency, Frequency, Monetary)"""
        
        self.logger.info("Criando features RFM")
        
        if 'created_at' not in df.columns or 'amount' not in df.columns:
            return df
        
        # Calcula RFM por usuário
        current_date = datetime.now()
        
        rfm_features = df.groupby('user_id').agg({
            'created_at': [
                lambda x: (current_date - x.max()).days,  # Recency
                'count'  # Frequency
            ],
            'amount': 'sum'  # Monetary
        })
        
        # Flatten columns
        rfm_features.columns = ['recency_days', 'frequency_transactions', 'monetary_value']
        
        # Normaliza e categoriza RFM
        for col in ['recency_days', 'frequency_transactions', 'monetary_value']:
            # Calcula quartis
            quartiles = np.percentile(rfm_features[col], [25, 50, 75])
            
            # Categoriza em scores 1-4
            def categorize_rfm(value, quartiles, reverse=False):
                if reverse:  # Para recency, menor é melhor
                    if value <= quartiles[0]:
                        return 4
                    elif value <= quartiles[1]:
                        return 3
                    elif value <= quartiles[2]:
                        return 2
                    else:
                        return 1
                else:
                    if value <= quartiles[0]:
                        return 1
                    elif value <= quartiles[1]:
                        return 2
                    elif value <= quartiles[2]:
                        return 3
                    else:
                        return 4
            
            reverse = col == 'recency_days'
            rfm_features[f'{col[:-5]}_score'] = rfm_features[col].apply(
                lambda x: categorize_rfm(x, quartiles, reverse)
            )
        
        # Score RFM combinado
        rfm_features['rfm_score'] = (
            rfm_features['recency_score'] * 100 +
            rfm_features['frequency_score'] * 10 +
            rfm_features['monetary_score']
        )
        
        # Segmentação RFM
        def categorize_rfm_segment(score):
            if score >= 444:
                return 'champions'
            elif score >= 334:
                return 'loyal_customers'
            elif score >= 224:
                return 'potential_loyalists'
            elif score >= 144:
                return 'at_risk'
            else:
                return 'hibernating'
        
        rfm_features['rfm_segment'] = rfm_features['rfm_score'].apply(categorize_rfm_segment)
        
        df = df.merge(rfm_features, on='user_id', how='left')
        
        report.feature_categories['rfm'] = len([col for col in df.columns if 'rfm' in col.lower()])
        
        return df
    
    def _create_segmentation_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Cria features específicas para segmentação"""
        
        self.logger.info("Criando features de segmentação")
        
        # 1. Score de Valor do Cliente (CLV proxy)
        clv_features = []
        if 'monetary_value' in df.columns:
            clv_features.append('monetary_value')
        if 'frequency_transactions' in df.columns:
            clv_features.append('frequency_transactions')
        if 'engagement_score' in df.columns:
            clv_features.append('engagement_score')
        
        if len(clv_features) >= 2:
            user_clv_data = df[['user_id'] + clv_features].drop_duplicates('user_id')
            
            # Normaliza features
            scaler = StandardScaler()
            clv_normalized = scaler.fit_transform(
                user_clv_data[clv_features].fillna(0)
            )
            
            user_clv_data['customer_lifetime_value_score'] = np.mean(clv_normalized, axis=1)
            
            df = df.merge(
                user_clv_data[['user_id', 'customer_lifetime_value_score']], 
                on='user_id', 
                how='left'
            )
        
        # 2. Score de Risco (Churn Risk)
        risk_features = []
        if 'recency_days' in df.columns:
            risk_features.append('recency_days')
        if 'activity_regularity' in df.columns:
            risk_features.append('activity_regularity')
        if 'bet_volatility' in df.columns:
            risk_features.append('bet_volatility')
        
        if len(risk_features) >= 2:
            user_risk_data = df[['user_id'] + risk_features].drop_duplicates('user_id')
            
            # Normaliza (valores altos = mais risco)
            scaler = StandardScaler()
            risk_normalized = scaler.fit_transform(
                user_risk_data[risk_features].fillna(0)
            )
            
            user_risk_data['churn_risk_score'] = np.mean(risk_normalized, axis=1)
            
            df = df.merge(
                user_risk_data[['user_id', 'churn_risk_score']], 
                on='user_id', 
                how='left'
            )
        
        # 3. Padrão de Comportamento (Behavioral Pattern)
        behavioral_features = []
        if 'game_diversity_score' in df.columns:
            behavioral_features.append('game_diversity_score')
        if 'weekend_activity_ratio' in df.columns:
            behavioral_features.append('weekend_activity_ratio')
        if 'device_variety' in df.columns:
            behavioral_features.append('device_variety')
        
        if behavioral_features:
            user_behavior_data = df[['user_id'] + behavioral_features].drop_duplicates('user_id')
            
            scaler = StandardScaler()
            behavior_normalized = scaler.fit_transform(
                user_behavior_data[behavioral_features].fillna(0)
            )
            
            user_behavior_data['behavioral_diversity_score'] = np.mean(behavior_normalized, axis=1)
            
            df = df.merge(
                user_behavior_data[['user_id', 'behavioral_diversity_score']], 
                on='user_id', 
                how='left'
            )
        
        report.feature_categories['segmentation'] = len([col for col in df.columns if any(
            keyword in col.lower() for keyword in ['score', 'risk', 'value', 'diversity']
        )])
        
        return df
    
    def _normalize_and_encode_features(self, df: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
        """Normalização e encoding de features"""
        
        self.logger.info("Aplicando normalização e encoding")
        
        # 1. Label Encoding para features categóricas
        categorical_features = [
            'favorite_game_type', 'canal_comunicacao_preferido', 'horarios_atividade',
            'ticket_medio_categoria', 'preferred_device', 'rfm_segment'
        ]
        
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self._label_encoders:
                    self._label_encoders[feature] = LabelEncoder()
                
                # Trata valores nulos
                df[feature] = df[feature].fillna('unknown')
                
                # Aplica encoding
                try:
                    df[f'{feature}_encoded'] = self._label_encoders[feature].fit_transform(df[feature])
                except Exception as e:
                    self.logger.warning(f"Erro no encoding de {feature}: {str(e)}")
        
        # 2. Normalização de features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns
        features_to_normalize = [col for col in numeric_features if not col.endswith('_encoded')]
        
        if features_to_normalize:
            # Aplica normalização apenas a features agregadas por usuário
            user_level_df = df.drop_duplicates('user_id')
            
            if len(user_level_df) > 1:  # Só normaliza se há múltiplos usuários
                normalized_data = self._scaler.fit_transform(
                    user_level_df[features_to_normalize].fillna(0)
                )
                
                normalized_df = pd.DataFrame(
                    normalized_data,
                    columns=[f'{col}_normalized' for col in features_to_normalize],
                    index=user_level_df.index
                )
                
                normalized_df['user_id'] = user_level_df['user_id'].values
                
                df = df.merge(normalized_df, on='user_id', how='left')
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula importância das features (método simples)"""
        
        numeric_features = df.select_dtypes(include=[np.number]).columns
        feature_importance = {}
        
        for feature in numeric_features:
            if feature != 'user_id':
                # Usa variância como proxy de importância
                variance = df[feature].var()
                feature_importance[feature] = variance
        
        # Normaliza scores
        if feature_importance:
            max_score = max(feature_importance.values())
            feature_importance = {
                k: v / max_score for k, v in feature_importance.items()
            }
        
        return feature_importance
    
    def create_ml_ready_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dataset final para ML"""
        
        self.logger.info("Preparando dataset para ML")
        
        # Remove features não necessárias para ML
        columns_to_remove = [
            'created_at', 'updated_at', '_source_file', '_extraction_timestamp'
        ]
        
        ml_df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        
        # Agrupa por usuário (uma linha por usuário)
        user_features = ml_df.drop_duplicates('user_id').copy()
        
        # Remove colunas com muitos valores únicos (não úteis para clustering)
        for col in user_features.columns:
            if user_features[col].dtype == 'object':
                unique_ratio = user_features[col].nunique() / len(user_features)
                if unique_ratio > 0.95:  # Mais de 95% valores únicos
                    user_features = user_features.drop(columns=[col])
        
        # Preenche valores nulos
        user_features = user_features.fillna(0)
        
        self.logger.info(
            "Dataset ML preparado",
            users=len(user_features),
            features=len(user_features.columns)
        )
        
        return user_features

# Função utilitária para teste
def test_feature_engineer():
    """Teste do feature engineer"""
    
    # Cria dados de teste
    test_data = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'],
        'amount': [100.0, 50.0, 200.0, 150.0, 75.0],
        'game_type': ['crash', 'cassino', 'esportes', 'crash', 'cassino'],
        'transaction_type': ['deposit', 'bet', 'deposit', 'bet', 'bet'],
        'created_at': [
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(hours=12),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(hours=6),
            datetime.now() - timedelta(days=3)
        ],
        'communication_channel': ['email', 'sms', 'email', 'push', 'email'],
        'device_type': ['mobile', 'desktop', 'mobile', 'mobile', 'desktop']
    })
    
    print("Dados originais:")
    print(test_data)
    print(f"Shape: {test_data.shape}")
    
    # Inicializa feature engineer
    engineer = FeatureEngineer()
    
    # Executa feature engineering
    engineered_data = engineer.engineer_features(test_data)
    
    print("\nDados com features engineered:")
    print(engineered_data.columns.tolist())
    print(f"Shape: {engineered_data.shape}")
    
    # Dataset para ML
    ml_dataset = engineer.create_ml_ready_dataset(engineered_data)
    print(f"\nDataset ML - Shape: {ml_dataset.shape}")
    print("Features finais:", ml_dataset.columns.tolist())

if __name__ == "__main__":
    test_feature_engineer()