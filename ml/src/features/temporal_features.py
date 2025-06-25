"""
Temporal Features Engineering - Extração de padrões temporais de usuários gaming/apostas
Implementação científica para identificar ciclos, sazonalidades e comportamentos temporais.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class TemporalFeatureEngine:
    """
    Engine para extração de features temporais de usuários gaming/apostas.
    
    Features extraídas:
    - Padrões circadianos (horários preferenciais)
    - Sazonalidade semanal/mensal
    - Ciclos de atividade e periodicidade
    - Timing de apostas e comportamentos
    - Mudanças temporais no comportamento
    """
    
    def __init__(self, 
                 timezone: str = 'UTC',
                 weekend_days: List[int] = [5, 6],  # Sábado e Domingo
                 business_hours: Tuple[int, int] = (9, 17),
                 peak_hours: Tuple[int, int] = (19, 23),
                 min_observations: int = 5):
        """
        Inicializa o engine de features temporais.
        
        Args:
            timezone: Timezone para análise temporal
            weekend_days: Dias da semana considerados fim de semana (0=Segunda)
            business_hours: Horário comercial (hora_inicio, hora_fim)
            peak_hours: Horário de pico (hora_inicio, hora_fim)
            min_observations: Mínimo de observações para calcular features
        """
        self.timezone = timezone
        self.weekend_days = weekend_days
        self.business_hours = business_hours
        self.peak_hours = peak_hours
        self.min_observations = min_observations
        
        # Caches para otimização
        self._feature_cache = {}
        self._computed_features = set()
        
    def extract_all_features(self, df: pd.DataFrame, 
                           timestamp_col: str = 'timestamp',
                           user_id_col: str = 'user_id') -> pd.DataFrame:
        """
        Extrai todas as features temporais.
        
        Args:
            df: DataFrame com dados de usuários e transações
            timestamp_col: Nome da coluna de timestamp
            user_id_col: Nome da coluna de user_id
            
        Returns:
            pd.DataFrame: DataFrame com features temporais
        """
        logger.info("Iniciando extração de features temporais...")
        
        # Validar dados de entrada
        self._validate_input_data(df, timestamp_col, user_id_col)
        
        # Preparar dados temporais
        df_processed = self._prepare_temporal_data(df, timestamp_col)
        
        # DataFrame para features
        user_ids = df_processed[user_id_col].unique() if user_id_col in df_processed.columns else df_processed.index
        features_df = pd.DataFrame(index=user_ids)
        
        # Agrupar por usuário se necessário
        if user_id_col in df_processed.columns:
            user_data = df_processed.groupby(user_id_col)
        else:
            # Assumir que cada linha é um usuário
            user_data = [(i, row) for i, row in df_processed.iterrows()]
        
        # Extrair features por categoria
        features_df = self._extract_circadian_features(user_data, features_df)
        features_df = self._extract_weekly_patterns(user_data, features_df)
        features_df = self._extract_activity_cycles(user_data, features_df)
        features_df = self._extract_timing_features(user_data, features_df)
        features_df = self._extract_temporal_evolution(user_data, features_df)
        features_df = self._extract_seasonal_features(user_data, features_df)
        
        logger.info(f"Extração concluída. {len(features_df.columns)} features temporais criadas")
        return features_df
    
    def _validate_input_data(self, df: pd.DataFrame, timestamp_col: str, user_id_col: str) -> None:
        """Valida se os dados de entrada têm as colunas necessárias."""
        if timestamp_col not in df.columns:
            raise ValueError(f"Coluna de timestamp '{timestamp_col}' não encontrada")
        
        if user_id_col not in df.columns and len(df) > 1000:
            logger.warning(f"Coluna de user_id '{user_id_col}' não encontrada. Assumindo dados agregados.")
    
    def _prepare_temporal_data(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Prepara dados temporais com features derivadas."""
        df_temp = df.copy()
        
        # Converter timestamp para datetime
        df_temp[timestamp_col] = pd.to_datetime(df_temp[timestamp_col])
        
        # Extrair componentes temporais
        df_temp['hour'] = df_temp[timestamp_col].dt.hour
        df_temp['day_of_week'] = df_temp[timestamp_col].dt.dayofweek
        df_temp['day_of_month'] = df_temp[timestamp_col].dt.day
        df_temp['month'] = df_temp[timestamp_col].dt.month
        df_temp['week_of_year'] = df_temp[timestamp_col].dt.isocalendar().week
        df_temp['quarter'] = df_temp[timestamp_col].dt.quarter
        
        # Features categóricas
        df_temp['is_weekend'] = df_temp['day_of_week'].isin(self.weekend_days)
        df_temp['is_business_hours'] = ((df_temp['hour'] >= self.business_hours[0]) & 
                                       (df_temp['hour'] <= self.business_hours[1]))
        df_temp['is_peak_hours'] = ((df_temp['hour'] >= self.peak_hours[0]) & 
                                   (df_temp['hour'] <= self.peak_hours[1]))
        
        # Períodos do dia
        df_temp['time_period'] = pd.cut(df_temp['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)
        
        return df_temp
    
    def _extract_circadian_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de padrões circadianos (24h)."""
        logger.info("Extraindo features circadianas...")
        
        circadian_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                circadian_features[user_id] = self._calculate_circadian_patterns(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    circadian_features[idx] = self._calculate_single_user_circadian(user_row)
        
        # Converter para DataFrame
        circadian_df = pd.DataFrame.from_dict(circadian_features, orient='index')
        
        # Combinar com features_df
        for col in circadian_df.columns:
            features_df[f'circadian_{col}'] = circadian_df[col]
        
        return features_df
    
    def _calculate_circadian_patterns(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula padrões circadianos para usuário."""
        profile = {}
        
        if len(user_df) < self.min_observations:
            return self._get_default_circadian_profile()
        
        hours = user_df['hour']
        
        # Estatísticas básicas de horário
        profile['preferred_hour'] = hours.mode().iloc[0] if not hours.mode().empty else hours.mean()
        profile['hour_mean'] = hours.mean()
        profile['hour_std'] = hours.std()
        
        # Concentração temporal (inverso da dispersão)
        hour_counts = hours.value_counts()
        hour_entropy = stats.entropy(hour_counts / len(hours))
        profile['temporal_concentration'] = 1 / (1 + hour_entropy)
        
        # Análise de periodicidade usando FFT
        try:
            # Criar series temporal por hora
            hourly_activity = user_df.groupby('hour').size()
            
            # Preencher horas ausentes com 0
            all_hours = pd.Series(index=range(24), data=0)
            all_hours.update(hourly_activity)
            
            # FFT para detectar periodicidade
            fft_values = fft(all_hours.values)
            fft_freqs = fftfreq(24, d=1)
            
            # Encontrar picos significativos
            power_spectrum = np.abs(fft_values)
            peaks, properties = find_peaks(power_spectrum[1:12], height=np.mean(power_spectrum) * 0.5)
            
            profile['circadian_strength'] = len(peaks) / 11  # Normalizado
            
            # Amplitude do pico principal (excluindo DC)
            if len(power_spectrum) > 1:
                profile['main_rhythm_amplitude'] = np.max(power_spectrum[1:]) / np.mean(power_spectrum)
            else:
                profile['main_rhythm_amplitude'] = 1
                
        except Exception:
            profile['circadian_strength'] = 0
            profile['main_rhythm_amplitude'] = 1
        
        # Padrões específicos
        profile['night_activity_ratio'] = ((hours >= 0) & (hours <= 6)).mean()
        profile['peak_hours_ratio'] = user_df['is_peak_hours'].mean()
        profile['business_hours_ratio'] = user_df['is_business_hours'].mean()
        
        # Regularidade (mesmo horário em dias diferentes)
        if 'day_of_week' in user_df.columns:
            hour_by_day = user_df.groupby('day_of_week')['hour'].mean()
            profile['daily_consistency'] = 1 / (1 + hour_by_day.std())
        
        return profile
    
    def _calculate_single_user_circadian(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula padrões circadianos para dados agregados."""
        profile = {}
        
        # Features baseadas em dados já agregados
        if 'preferred_hour' in user_row:
            preferred_hour = user_row.get('preferred_hour', 12)
            profile['preferred_hour'] = preferred_hour
            
            # Classificar período do dia
            if 0 <= preferred_hour <= 6:
                profile['night_activity_ratio'] = 1
                profile['peak_hours_ratio'] = 0
                profile['business_hours_ratio'] = 0
            elif 7 <= preferred_hour <= 11:
                profile['night_activity_ratio'] = 0
                profile['peak_hours_ratio'] = 0
                profile['business_hours_ratio'] = 1
            elif 12 <= preferred_hour <= 17:
                profile['night_activity_ratio'] = 0
                profile['peak_hours_ratio'] = 0
                profile['business_hours_ratio'] = 1
            elif 18 <= preferred_hour <= 23:
                profile['night_activity_ratio'] = 0
                profile['peak_hours_ratio'] = 1
                profile['business_hours_ratio'] = 0
        
        return profile
    
    def _get_default_circadian_profile(self) -> Dict[str, float]:
        """Retorna perfil circadiano padrão para usuários com poucos dados."""
        return {
            'preferred_hour': 20,
            'hour_mean': 20,
            'hour_std': 4,
            'temporal_concentration': 0.5,
            'circadian_strength': 0.3,
            'main_rhythm_amplitude': 1.5,
            'night_activity_ratio': 0.1,
            'peak_hours_ratio': 0.6,
            'business_hours_ratio': 0.3,
            'daily_consistency': 0.7
        }
    
    def _extract_weekly_patterns(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai padrões semanais."""
        logger.info("Extraindo padrões semanais...")
        
        weekly_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                weekly_features[user_id] = self._calculate_weekly_patterns(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    weekly_features[idx] = self._calculate_single_user_weekly(user_row)
        
        # Converter para DataFrame
        weekly_df = pd.DataFrame.from_dict(weekly_features, orient='index')
        
        # Combinar com features_df
        for col in weekly_df.columns:
            features_df[f'weekly_{col}'] = weekly_df[col]
        
        return features_df
    
    def _calculate_weekly_patterns(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula padrões semanais."""
        profile = {}
        
        if len(user_df) < self.min_observations:
            return self._get_default_weekly_profile()
        
        # Análise por dia da semana
        day_counts = user_df['day_of_week'].value_counts()
        total_days = len(user_df)
        
        # Weekend vs Weekday behavior
        weekend_activity = user_df['is_weekend'].mean()
        profile['weekend_activity_ratio'] = weekend_activity
        
        # Distribuição de atividade por dia
        day_distribution = day_counts / total_days
        day_entropy = stats.entropy(day_distribution)
        profile['weekly_diversity'] = day_entropy / np.log(7)  # Normalizado
        
        # Concentração em dias específicos
        profile['max_day_concentration'] = day_distribution.max()
        
        # Padrão de fim de semana
        if len(day_counts) >= 7:
            weekend_days_activity = day_counts[day_counts.index.isin(self.weekend_days)]
            weekday_activity = day_counts[~day_counts.index.isin(self.weekend_days)]
            
            if len(weekday_activity) > 0:
                profile['weekend_vs_weekday_ratio'] = weekend_days_activity.mean() / weekday_activity.mean()
            else:
                profile['weekend_vs_weekday_ratio'] = 1
        else:
            profile['weekend_vs_weekday_ratio'] = 1
        
        # Consistência semanal
        if 'week_of_year' in user_df.columns:
            weekly_activity = user_df.groupby('week_of_year').size()
            if len(weekly_activity) > 1:
                profile['weekly_consistency'] = 1 / (1 + weekly_activity.std() / weekly_activity.mean())
            else:
                profile['weekly_consistency'] = 1
        
        return profile
    
    def _calculate_single_user_weekly(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula padrões semanais para dados agregados."""
        profile = {}
        
        # Inferir padrões baseado no horário preferido e outros indicadores
        if 'preferred_hour' in user_row:
            preferred_hour = user_row.get('preferred_hour', 20)
            
            # Horários tardios podem indicar atividade de fim de semana
            if preferred_hour >= 22 or preferred_hour <= 2:
                profile['weekend_activity_ratio'] = 0.7
            else:
                profile['weekend_activity_ratio'] = 0.3
        
        # Se temos dados de frequência, inferir distribuição semanal
        if 'session_frequency' in user_row:
            frequency = user_row.get('session_frequency', 7)
            # Maior frequência pode indicar distribuição mais uniforme
            profile['weekly_diversity'] = min(frequency / 21, 1)  # Normalizado
        
        return profile
    
    def _get_default_weekly_profile(self) -> Dict[str, float]:
        """Retorna perfil semanal padrão."""
        return {
            'weekend_activity_ratio': 0.4,
            'weekly_diversity': 0.6,
            'max_day_concentration': 0.25,
            'weekend_vs_weekday_ratio': 1.2,
            'weekly_consistency': 0.7
        }
    
    def _extract_activity_cycles(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai ciclos de atividade."""
        logger.info("Extraindo ciclos de atividade...")
        
        cycle_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                cycle_features[user_id] = self._calculate_activity_cycles(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    cycle_features[idx] = self._calculate_single_user_cycles(user_row)
        
        # Converter para DataFrame
        cycle_df = pd.DataFrame.from_dict(cycle_features, orient='index')
        
        # Combinar com features_df
        for col in cycle_df.columns:
            features_df[f'cycle_{col}'] = cycle_df[col]
        
        return features_df
    
    def _calculate_activity_cycles(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula ciclos de atividade."""
        profile = {}
        
        if len(user_df) < self.min_observations:
            return self._get_default_cycle_profile()
        
        # Ordenar por timestamp
        user_df_sorted = user_df.sort_values('timestamp')
        
        # Calcular intervalos entre sessões
        time_diffs = user_df_sorted['timestamp'].diff()
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600
        time_diffs_clean = time_diffs_hours.dropna()
        
        if len(time_diffs_clean) > 2:
            # Estatísticas de intervalo
            profile['avg_session_interval'] = time_diffs_clean.mean()
            profile['session_interval_cv'] = time_diffs_clean.std() / time_diffs_clean.mean()
            
            # Detectar periodicidade nos intervalos
            try:
                # Usar FFT nos intervalos para detectar ciclos
                if len(time_diffs_clean) >= 8:
                    # Interpolar para grid regular se necessário
                    fft_input = time_diffs_clean.values
                    fft_values = fft(fft_input)
                    power_spectrum = np.abs(fft_values)
                    
                    # Encontrar picos principais
                    peaks, _ = find_peaks(power_spectrum[1:len(power_spectrum)//2], 
                                        height=np.mean(power_spectrum) * 0.3)
                    
                    profile['cycle_regularity'] = len(peaks) / (len(power_spectrum)//2 - 1)
                else:
                    profile['cycle_regularity'] = 0.5
                    
            except Exception:
                profile['cycle_regularity'] = 0.5
        
        # Atividade burst vs sustained
        # Definir bursts como períodos com intervalo < 2 horas
        if len(time_diffs_clean) > 0:
            burst_sessions = (time_diffs_clean < 2).sum()
            profile['burst_activity_ratio'] = burst_sessions / len(time_diffs_clean)
        
        # Detectar abandono/retorno patterns
        if len(time_diffs_clean) > 5:
            # Intervalos longos (> 7 dias) como indicadores de abandono temporário
            long_breaks = (time_diffs_clean > 168).sum()  # 7 dias = 168 horas
            profile['long_break_frequency'] = long_breaks / len(time_diffs_clean)
        
        return profile
    
    def _calculate_single_user_cycles(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula ciclos para dados agregados."""
        profile = {}
        
        # Aproximações baseadas em dados disponíveis
        if 'session_frequency' in user_row:
            frequency = user_row.get('session_frequency', 1)
            # Maior frequência sugere ciclos mais curtos
            profile['avg_session_interval'] = 168 / max(frequency, 1)  # horas por semana
            profile['cycle_regularity'] = min(frequency / 14, 1)  # Normalizado
        
        if 'days_since_last_bet' in user_row:
            days_since = user_row.get('days_since_last_bet', 1)
            # Recência como indicador de ciclo ativo
            profile['cycle_activity'] = 1 / (1 + days_since / 7)
        
        return profile
    
    def _get_default_cycle_profile(self) -> Dict[str, float]:
        """Retorna perfil de ciclo padrão."""
        return {
            'avg_session_interval': 48,
            'session_interval_cv': 1.5,
            'cycle_regularity': 0.4,
            'burst_activity_ratio': 0.2,
            'long_break_frequency': 0.1
        }
    
    def _extract_timing_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de timing específico."""
        logger.info("Extraindo features de timing...")
        
        timing_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                timing_features[user_id] = self._calculate_timing_features(user_df)
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
    
    def _calculate_timing_features(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula features específicas de timing."""
        profile = {}
        
        # Duração das sessões vs horário
        if 'session_duration' in user_df.columns:
            duration_by_hour = user_df.groupby('hour')['session_duration'].mean()
            
            if len(duration_by_hour) > 1:
                # Variabilidade da duração por horário
                profile['duration_by_hour_cv'] = duration_by_hour.std() / duration_by_hour.mean()
                
                # Horário de sessões mais longas
                profile['longest_session_hour'] = duration_by_hour.idxmax()
        
        # Comportamento de aposta vs timing
        if 'bet_amount' in user_df.columns:
            bet_by_hour = user_df.groupby('hour')['bet_amount'].mean()
            
            if len(bet_by_hour) > 1:
                # Correlação entre horário e valor da aposta
                hours = bet_by_hour.index
                bets = bet_by_hour.values
                
                correlation, p_value = stats.spearmanr(hours, bets)
                profile['bet_hour_correlation'] = correlation
                profile['bet_timing_significance'] = 1 - p_value
        
        # Primeiro vs último horário do dia
        if len(user_df) > 1:
            daily_first_last = user_df.groupby(user_df['timestamp'].dt.date)['hour'].agg(['min', 'max'])
            
            if len(daily_first_last) > 0:
                profile['daily_hour_span'] = (daily_first_last['max'] - daily_first_last['min']).mean()
                profile['early_starter'] = (daily_first_last['min'] <= 8).mean()
                profile['late_finisher'] = (daily_first_last['max'] >= 22).mean()
        
        return profile
    
    def _calculate_single_user_timing(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula timing para dados agregados."""
        profile = {}
        
        if 'preferred_hour' in user_row:
            preferred_hour = user_row.get('preferred_hour', 20)
            
            # Classificações de timing
            profile['early_starter'] = 1 if preferred_hour <= 8 else 0
            profile['late_finisher'] = 1 if preferred_hour >= 22 else 0
            profile['longest_session_hour'] = preferred_hour
        
        return profile
    
    def _extract_temporal_evolution(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai evolução temporal do comportamento."""
        logger.info("Extraindo evolução temporal...")
        
        evolution_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                evolution_features[user_id] = self._calculate_temporal_evolution(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    evolution_features[idx] = self._calculate_single_user_evolution(user_row)
        
        # Converter para DataFrame
        evolution_df = pd.DataFrame.from_dict(evolution_features, orient='index')
        
        # Combinar com features_df
        for col in evolution_df.columns:
            features_df[f'evolution_{col}'] = evolution_df[col]
        
        return features_df
    
    def _calculate_temporal_evolution(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula evolução temporal."""
        profile = {}
        
        if len(user_df) < self.min_observations:
            return {}
        
        # Ordenar por timestamp
        user_df_sorted = user_df.sort_values('timestamp')
        
        # Trend na atividade (primeira vs segunda metade)
        mid_point = len(user_df_sorted) // 2
        first_half = user_df_sorted.iloc[:mid_point]
        second_half = user_df_sorted.iloc[mid_point:]
        
        if len(first_half) > 0 and len(second_half) > 0:
            # Mudança na frequência
            first_span = (first_half['timestamp'].max() - first_half['timestamp'].min()).total_seconds() / 86400  # dias
            second_span = (second_half['timestamp'].max() - second_half['timestamp'].min()).total_seconds() / 86400
            
            if first_span > 0 and second_span > 0:
                first_frequency = len(first_half) / first_span
                second_frequency = len(second_half) / second_span
                
                profile['frequency_trend'] = (second_frequency - first_frequency) / first_frequency
        
        # Tendência temporal em features numéricas
        numeric_cols = user_df_sorted.select_dtypes(include=[np.number]).columns
        
        for col in ['session_duration', 'bet_amount', 'games_played']:
            if col in numeric_cols:
                values = user_df_sorted[col].dropna()
                
                if len(values) >= 3:
                    x = np.arange(len(values))
                    slope, _, r_value, p_value, _ = stats.linregress(x, values)
                    
                    profile[f'{col}_trend'] = slope / (values.mean() + 1e-8)  # Normalizado
                    profile[f'{col}_trend_strength'] = r_value ** 2
        
        return profile
    
    def _calculate_single_user_evolution(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula evolução para dados agregados."""
        profile = {}
        
        # Usar recência como proxy de evolução
        if 'days_since_last_bet' in user_row:
            days_since = user_row.get('days_since_last_bet', 1)
            # Usuários recentes podem estar em fase ativa
            profile['activity_recency'] = 1 / (1 + days_since / 30)
        
        return profile
    
    def _extract_seasonal_features(self, user_data: Any, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features sazonais."""
        logger.info("Extraindo features sazonais...")
        
        seasonal_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                seasonal_features[user_id] = self._calculate_seasonal_patterns(user_df)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    seasonal_features[idx] = self._calculate_single_user_seasonal(user_row)
        
        # Converter para DataFrame
        seasonal_df = pd.DataFrame.from_dict(seasonal_features, orient='index')
        
        # Combinar com features_df
        for col in seasonal_df.columns:
            features_df[f'seasonal_{col}'] = seasonal_df[col]
        
        return features_df
    
    def _calculate_seasonal_patterns(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula padrões sazonais."""
        profile = {}
        
        if len(user_df) < self.min_observations:
            return {}
        
        # Padrões mensais
        if 'month' in user_df.columns:
            month_counts = user_df['month'].value_counts()
            
            if len(month_counts) > 1:
                # Sazonalidade mensal
                month_entropy = stats.entropy(month_counts / len(user_df))
                profile['monthly_seasonality'] = 1 - (month_entropy / np.log(12))
        
        # Padrões trimestrais
        if 'quarter' in user_df.columns:
            quarter_counts = user_df['quarter'].value_counts()
            
            if len(quarter_counts) > 1:
                # Distribuição por trimestre
                quarter_dist = quarter_counts / len(user_df)
                profile['quarter_concentration'] = quarter_dist.max()
        
        return profile
    
    def _calculate_single_user_seasonal(self, user_row: pd.Series) -> Dict[str, float]:
        """Calcula sazonalidade para dados agregados."""
        # Para dados agregados, assumir distribuição uniforme
        return {
            'monthly_seasonality': 0.5,
            'quarter_concentration': 0.25
        }
    
    def get_temporal_insights(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights das features temporais extraídas.
        
        Args:
            features_df: DataFrame com features temporais
            
        Returns:
            Dict com insights temporais
        """
        insights = {}
        
        # Análise de horários preferenciais
        if 'circadian_preferred_hour' in features_df.columns:
            preferred_hours = features_df['circadian_preferred_hour'].dropna()
            
            insights['peak_usage_hours'] = {
                'most_common': preferred_hours.mode().tolist(),
                'mean': preferred_hours.mean(),
                'distribution': preferred_hours.value_counts().head(5).to_dict()
            }
        
        # Padrões de fim de semana
        if 'weekly_weekend_activity_ratio' in features_df.columns:
            weekend_ratios = features_df['weekly_weekend_activity_ratio'].dropna()
            
            insights['weekend_behavior'] = {
                'weekend_heavy_users': (weekend_ratios > 0.6).mean(),
                'weekday_only_users': (weekend_ratios < 0.2).mean(),
                'balanced_users': ((weekend_ratios >= 0.4) & (weekend_ratios <= 0.6)).mean()
            }
        
        # Consistência temporal
        temporal_consistency_cols = [col for col in features_df.columns if 'consistency' in col]
        if temporal_consistency_cols:
            consistency_scores = features_df[temporal_consistency_cols].mean(axis=1)
            
            insights['temporal_consistency'] = {
                'highly_consistent': (consistency_scores > 0.8).mean(),
                'moderately_consistent': ((consistency_scores >= 0.5) & (consistency_scores <= 0.8)).mean(),
                'inconsistent': (consistency_scores < 0.5).mean()
            }
        
        return insights
    
    def export_temporal_documentation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exporta documentação das features temporais.
        
        Args:
            features_df: DataFrame com features
            
        Returns:
            pd.DataFrame: Documentação das features
        """
        documentation = []
        
        feature_descriptions = {
            # Circadian features
            'circadian_preferred_hour': 'Horário preferencial do usuário (0-23)',
            'circadian_temporal_concentration': 'Concentração da atividade em horários específicos',
            'circadian_night_activity_ratio': 'Proporção de atividade noturna (0-6h)',
            'circadian_peak_hours_ratio': 'Proporção de atividade em horário de pico',
            'circadian_daily_consistency': 'Consistência de horários entre dias',
            
            # Weekly features
            'weekly_weekend_activity_ratio': 'Proporção de atividade em fins de semana',
            'weekly_diversity': 'Diversidade de atividade ao longo da semana',
            'weekly_consistency': 'Consistência de atividade semanal',
            
            # Cycle features
            'cycle_avg_session_interval': 'Intervalo médio entre sessões (horas)',
            'cycle_regularity': 'Regularidade dos ciclos de atividade',
            'cycle_burst_activity_ratio': 'Proporção de atividade em rajadas',
            
            # Timing features
            'timing_early_starter': 'Indicador de início precoce das atividades',
            'timing_late_finisher': 'Indicador de término tardio das atividades',
            'timing_bet_hour_correlation': 'Correlação entre horário e valor das apostas',
            
            # Evolution features
            'evolution_frequency_trend': 'Tendência na frequência de atividade',
            'evolution_activity_recency': 'Score baseado na recência da atividade',
            
            # Seasonal features
            'seasonal_monthly_seasonality': 'Sazonalidade mensal da atividade',
            'seasonal_quarter_concentration': 'Concentração de atividade por trimestre'
        }
        
        for feature in features_df.columns:
            doc_entry = {
                'feature_name': feature,
                'category': feature.split('_')[0] if '_' in feature else 'temporal',
                'description': feature_descriptions.get(feature, 'Feature temporal'),
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
    
    # Simular dados temporais de usuários
    n_users = 300
    n_days = 90
    
    test_data = []
    
    for user_id in range(n_users):
        # Diferentes perfis temporais
        if user_id < 100:  # Usuários noturnos regulares
            preferred_hours = [22, 23, 0, 1]
            preferred_days = [4, 5, 6]  # Sexta, sábado, domingo
            session_pattern = 'regular'
        elif user_id < 200:  # Usuários diurnos casuais
            preferred_hours = [12, 13, 14, 15, 16]
            preferred_days = [0, 1, 2, 3, 4]  # Dias da semana
            session_pattern = 'casual'
        else:  # Usuários irregulares
            preferred_hours = list(range(24))
            preferred_days = list(range(7))
            session_pattern = 'irregular'
        
        # Gerar sessões para este usuário
        n_sessions = np.random.poisson(20) + 5
        
        for session in range(n_sessions):
            # Data aleatória nos últimos 90 dias
            days_ago = np.random.randint(0, n_days)
            base_date = datetime.now() - timedelta(days=days_ago)
            
            # Ajustar por padrão do usuário
            if session_pattern == 'regular':
                # Preferencialmente nos dias/horários favoritos
                hour = np.random.choice(preferred_hours)
                day_adjustment = np.random.choice([-2, -1, 0, 1, 2])
                target_weekday = np.random.choice(preferred_days)
                
                # Ajustar data para cair no dia da semana desejado
                current_weekday = base_date.weekday()
                days_to_target = (target_weekday - current_weekday) % 7
                adjusted_date = base_date + timedelta(days=days_to_target + day_adjustment)
                
            elif session_pattern == 'casual':
                hour = np.random.choice(preferred_hours)
                adjusted_date = base_date
                
            else:  # irregular
                hour = np.random.choice(preferred_hours)
                adjusted_date = base_date
            
            timestamp = adjusted_date.replace(hour=hour, minute=np.random.randint(0, 60))
            
            # Dados da sessão
            session_duration = np.random.exponential(30) + 5
            games_played = np.random.poisson(5) + 1
            bet_amount = np.random.lognormal(2, 1)
            
            test_data.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'session_duration': session_duration,
                'games_played': games_played,
                'bet_amount': bet_amount
            })
    
    test_df = pd.DataFrame(test_data)
    
    # Testar extração de features temporais
    print("🚀 Testando Temporal Feature Engine...")
    
    engine = TemporalFeatureEngine()
    temporal_features = engine.extract_all_features(test_df)
    
    print(f"✅ Features temporais extraídas: {len(temporal_features.columns)}")
    print(f"✅ Usuários processados: {len(temporal_features)}")
    
    # Mostrar algumas features
    print(f"\n✅ Primeiras features temporais:")
    print(temporal_features.head())
    
    # Insights temporais
    insights = engine.get_temporal_insights(temporal_features)
    
    if 'peak_usage_hours' in insights:
        print(f"\n✅ Horários de pico mais comuns: {insights['peak_usage_hours']['most_common']}")
    
    if 'weekend_behavior' in insights:
        weekend_stats = insights['weekend_behavior']
        print(f"✅ Usuários focados em fim de semana: {weekend_stats['weekend_heavy_users']:.1%}")
        print(f"✅ Usuários apenas dias úteis: {weekend_stats['weekday_only_users']:.1%}")
    
    # Documentação
    documentation = engine.export_temporal_documentation(temporal_features)
    print(f"\n✅ Documentação gerada para {len(documentation)} features temporais")
    
    print("\n🎯 Temporal Feature Engine implementado com sucesso!")