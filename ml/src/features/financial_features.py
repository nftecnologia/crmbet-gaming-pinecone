"""
Financial Features Engineering - Extração de padrões financeiros de usuários gaming/apostas
Implementação científica para analisar comportamentos monetários e gestão de bankroll.

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

class FinancialFeatureEngine:
    """
    Engine para extração de features financeiras de usuários gaming/apostas.
    
    Features extraídas:
    - Padrões de ticket médio e distribuição
    - Gestão de bankroll e disciplina financeira
    - Comportamento de depósito e saque
    - Análise de volatilidade financeira
    - Eficiência e ROI patterns
    - Comportamento pós ganhos/perdas
    """
    
    def __init__(self, 
                 currency_symbol: str = '$',
                 high_roller_threshold: float = 1000,
                 whale_threshold: float = 10000,
                 loss_chase_window: int = 3,
                 volatility_window: int = 10):
        """
        Inicializa o engine de features financeiras.
        
        Args:
            currency_symbol: Símbolo da moeda para formatação
            high_roller_threshold: Threshold para classificar high roller
            whale_threshold: Threshold para classificar whale
            loss_chase_window: Janela para detectar chase de perdas
            volatility_window: Janela para calcular volatilidade
        """
        self.currency_symbol = currency_symbol
        self.high_roller_threshold = high_roller_threshold
        self.whale_threshold = whale_threshold
        self.loss_chase_window = loss_chase_window
        self.volatility_window = volatility_window
        
        # Caches para otimização
        self._feature_cache = {}
        self._computed_features = set()
        
    def extract_all_features(self, df: pd.DataFrame, 
                           user_id_col: str = 'user_id',
                           amount_cols: Dict[str, str] = None) -> pd.DataFrame:
        """
        Extrai todas as features financeiras.
        
        Args:
            df: DataFrame com dados financeiros dos usuários
            user_id_col: Nome da coluna de user_id
            amount_cols: Mapeamento de colunas financeiras
                        {'bet': 'bet_amount', 'win': 'win_amount', 'deposit': 'deposit_amount'}
            
        Returns:
            pd.DataFrame: DataFrame com features financeiras
        """
        logger.info("Iniciando extração de features financeiras...")
        
        # Definir colunas padrão se não fornecidas
        if amount_cols is None:
            amount_cols = {
                'bet': 'bet_amount',
                'win': 'win_amount', 
                'deposit': 'deposit_amount',
                'withdrawal': 'withdrawal_amount'
            }
        
        # Validar dados de entrada
        self._validate_input_data(df, user_id_col, amount_cols)
        
        # DataFrame para features
        user_ids = df[user_id_col].unique() if user_id_col in df.columns else df.index
        features_df = pd.DataFrame(index=user_ids)
        
        # Agrupar por usuário se necessário
        if user_id_col in df.columns:
            user_data = df.groupby(user_id_col)
        else:
            # Assumir que cada linha é um usuário
            user_data = [(i, row) for i, row in df.iterrows()]
        
        # Extrair features por categoria
        features_df = self._extract_spending_features(user_data, features_df, amount_cols)
        features_df = self._extract_bankroll_features(user_data, features_df, amount_cols)
        features_df = self._extract_volatility_features(user_data, features_df, amount_cols)
        features_df = self._extract_profitability_features(user_data, features_df, amount_cols)
        features_df = self._extract_transaction_features(user_data, features_df, amount_cols)
        features_df = self._extract_risk_management_features(user_data, features_df, amount_cols)
        
        logger.info(f"Extração concluída. {len(features_df.columns)} features financeiras criadas")
        return features_df
    
    def _validate_input_data(self, df: pd.DataFrame, user_id_col: str, amount_cols: Dict[str, str]) -> None:
        """Valida se os dados de entrada têm as colunas necessárias."""
        missing_cols = []
        
        if user_id_col not in df.columns and len(df) > 1000:
            logger.warning(f"Coluna de user_id '{user_id_col}' não encontrada. Assumindo dados agregados.")
        
        # Verificar pelo menos uma coluna de valores
        available_amount_cols = [col for col in amount_cols.values() if col in df.columns]
        if not available_amount_cols:
            logger.warning("Nenhuma coluna de valores financeiros encontrada. Verificar mapeamento.")
    
    def _extract_spending_features(self, user_data: Any, features_df: pd.DataFrame, 
                                 amount_cols: Dict[str, str]) -> pd.DataFrame:
        """Extrai features de padrões de gastos."""
        logger.info("Extraindo features de padrões de gastos...")
        
        spending_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                spending_features[user_id] = self._calculate_user_spending(user_df, amount_cols)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    spending_features[idx] = self._calculate_single_user_spending(user_row, amount_cols)
        
        # Converter para DataFrame
        spending_df = pd.DataFrame.from_dict(spending_features, orient='index')
        
        # Combinar com features_df
        for col in spending_df.columns:
            features_df[f'spending_{col}'] = spending_df[col]
        
        return features_df
    
    def _calculate_user_spending(self, user_df: pd.DataFrame, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula padrões de gastos para usuário."""
        profile = {}
        
        bet_col = amount_cols.get('bet', 'bet_amount')
        
        if bet_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            
            if len(bets) > 0:
                # Estatísticas básicas
                profile['total_wagered'] = bets.sum()
                profile['avg_bet'] = bets.mean()
                profile['median_bet'] = bets.median()
                profile['max_bet'] = bets.max()
                profile['min_bet'] = bets.min()
                
                # Distribuição e dispersão
                profile['bet_std'] = bets.std()
                profile['bet_cv'] = bets.std() / (bets.mean() + 1e-8)
                profile['bet_skewness'] = stats.skew(bets)
                profile['bet_kurtosis'] = stats.kurtosis(bets)
                
                # Percentis
                profile['bet_p25'] = bets.quantile(0.25)
                profile['bet_p75'] = bets.quantile(0.75)
                profile['bet_p90'] = bets.quantile(0.90)
                profile['bet_p95'] = bets.quantile(0.95)
                
                # Ratios importantes
                profile['max_bet_ratio'] = profile['max_bet'] / (profile['avg_bet'] + 1e-8)
                profile['p95_avg_ratio'] = profile['bet_p95'] / (profile['avg_bet'] + 1e-8)
                
                # Classificação de usuário
                if profile['avg_bet'] >= self.whale_threshold:
                    profile['user_tier'] = 3  # Whale
                elif profile['avg_bet'] >= self.high_roller_threshold:
                    profile['user_tier'] = 2  # High Roller
                else:
                    profile['user_tier'] = 1  # Regular
                
                # Análise de concentração de gastos
                bet_counts = pd.cut(bets, bins=5).value_counts()
                bet_distribution = bet_counts / len(bets)
                profile['spending_concentration'] = (bet_distribution ** 2).sum()  # Herfindahl index
        
        return profile
    
    def _calculate_single_user_spending(self, user_row: pd.Series, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula gastos para dados agregados."""
        profile = {}
        
        # Mapear colunas possíveis
        bet_cols = ['avg_bet_amount', 'total_bet_amount', 'bet_amount', 'average_bet', 'total_wagered']
        found_bet_col = None
        
        for col in bet_cols:
            if col in user_row.index:
                found_bet_col = col
                break
        
        if found_bet_col:
            avg_bet = user_row.get(found_bet_col, 0)
            profile['avg_bet'] = avg_bet
            
            # Inferir outras métricas baseado no que está disponível
            if 'total_bet_amount' in user_row.index:
                profile['total_wagered'] = user_row.get('total_bet_amount', 0)
            
            if 'max_bet_amount' in user_row.index:
                max_bet = user_row.get('max_bet_amount', avg_bet)
                profile['max_bet'] = max_bet
                profile['max_bet_ratio'] = max_bet / (avg_bet + 1e-8)
            
            # Classificação
            if avg_bet >= self.whale_threshold:
                profile['user_tier'] = 3
            elif avg_bet >= self.high_roller_threshold:
                profile['user_tier'] = 2
            else:
                profile['user_tier'] = 1
        
        return profile
    
    def _extract_bankroll_features(self, user_data: Any, features_df: pd.DataFrame, 
                                 amount_cols: Dict[str, str]) -> pd.DataFrame:
        """Extrai features de gestão de bankroll."""
        logger.info("Extraindo features de gestão de bankroll...")
        
        bankroll_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                bankroll_features[user_id] = self._calculate_bankroll_management(user_df, amount_cols)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    bankroll_features[idx] = self._calculate_single_user_bankroll(user_row, amount_cols)
        
        # Converter para DataFrame
        bankroll_df = pd.DataFrame.from_dict(bankroll_features, orient='index')
        
        # Combinar com features_df
        for col in bankroll_df.columns:
            features_df[f'bankroll_{col}'] = bankroll_df[col]
        
        return features_df
    
    def _calculate_bankroll_management(self, user_df: pd.DataFrame, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula gestão de bankroll."""
        profile = {}
        
        bet_col = amount_cols.get('bet', 'bet_amount')
        deposit_col = amount_cols.get('deposit', 'deposit_amount')
        
        # Análise de depósitos
        if deposit_col in user_df.columns:
            deposits = user_df[deposit_col].dropna()
            deposits = deposits[deposits > 0]  # Apenas depósitos positivos
            
            if len(deposits) > 0:
                profile['total_deposits'] = deposits.sum()
                profile['avg_deposit'] = deposits.mean()
                profile['deposit_frequency'] = len(deposits)
                profile['max_deposit'] = deposits.max()
                
                # Consistência dos depósitos
                profile['deposit_cv'] = deposits.std() / (deposits.mean() + 1e-8)
        
        # Relação bet/deposit (gestão de risco)
        if bet_col in user_df.columns and deposit_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            deposits = user_df[deposit_col].dropna()
            deposits = deposits[deposits > 0]
            
            if len(bets) > 0 and len(deposits) > 0:
                # Ratio de apostas vs depósitos
                total_bets = bets.sum()
                total_deposits = deposits.sum()
                
                profile['bet_deposit_ratio'] = total_bets / (total_deposits + 1e-8)
                
                # Análise de sizing (aposta vs bankroll disponível)
                avg_bet = bets.mean()
                avg_deposit = deposits.mean()
                profile['bet_sizing_ratio'] = avg_bet / (avg_deposit + 1e-8)
                
                # Kelly criterion approximation (se temos win rate)
                if 'win_rate' in user_df.columns:
                    win_rate = user_df['win_rate'].mean()
                    if 0 < win_rate < 1:
                        # Simplified Kelly (assuming even money bets)
                        kelly_f = (2 * win_rate - 1)
                        current_sizing = avg_bet / (avg_deposit + 1e-8)
                        profile['kelly_deviation'] = abs(current_sizing - max(0, kelly_f))
        
        # Disciplina financeira (consistência no sizing)
        if bet_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            
            if len(bets) > 2:
                # Analisar mudanças no sizing
                bet_changes = bets.pct_change().dropna()
                
                if len(bet_changes) > 0:
                    profile['sizing_volatility'] = bet_changes.std()
                    profile['sizing_discipline'] = 1 / (1 + profile['sizing_volatility'])
        
        return profile
    
    def _calculate_single_user_bankroll(self, user_row: pd.Series, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula bankroll para dados agregados."""
        profile = {}
        
        # Mapear colunas de depósitos
        deposit_cols = ['total_deposits', 'avg_deposit', 'deposit_amount']
        for col in deposit_cols:
            if col in user_row.index:
                profile['total_deposits'] = user_row.get(col, 0)
                break
        
        # Bet/deposit ratio se disponível
        if 'avg_bet_amount' in user_row.index and 'total_deposits' in profile:
            avg_bet = user_row.get('avg_bet_amount', 0)
            total_deposits = profile['total_deposits']
            
            if total_deposits > 0:
                profile['bet_sizing_ratio'] = avg_bet / total_deposits
        
        return profile
    
    def _extract_volatility_features(self, user_data: Any, features_df: pd.DataFrame, 
                                   amount_cols: Dict[str, str]) -> pd.DataFrame:
        """Extrai features de volatilidade financeira."""
        logger.info("Extraindo features de volatilidade financeira...")
        
        volatility_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                volatility_features[user_id] = self._calculate_financial_volatility(user_df, amount_cols)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    volatility_features[idx] = self._calculate_single_user_volatility(user_row, amount_cols)
        
        # Converter para DataFrame
        volatility_df = pd.DataFrame.from_dict(volatility_features, orient='index')
        
        # Combinar com features_df
        for col in volatility_df.columns:
            features_df[f'volatility_{col}'] = volatility_df[col]
        
        return features_df
    
    def _calculate_financial_volatility(self, user_df: pd.DataFrame, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula volatilidade financeira."""
        profile = {}
        
        bet_col = amount_cols.get('bet', 'bet_amount')
        
        if bet_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            
            if len(bets) >= self.volatility_window:
                # Volatilidade rolling
                bet_series = pd.Series(bets.values)
                rolling_vol = bet_series.rolling(window=self.volatility_window).std()
                
                profile['avg_volatility'] = rolling_vol.mean()
                profile['max_volatility'] = rolling_vol.max()
                profile['volatility_trend'] = self._calculate_trend(rolling_vol.dropna().values)
                
                # Volatilidade normalizada
                avg_bet = bets.mean()
                profile['normalized_volatility'] = rolling_vol.mean() / (avg_bet + 1e-8)
        
        # Volatilidade de P&L se temos wins
        win_col = amount_cols.get('win', 'win_amount')
        if bet_col in user_df.columns and win_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            wins = user_df[win_col].fillna(0)
            
            if len(bets) > 0 and len(wins) > 0:
                # Calcular P&L por sessão
                pnl = wins - bets
                
                if len(pnl) >= self.volatility_window:
                    pnl_series = pd.Series(pnl.values)
                    rolling_pnl_vol = pnl_series.rolling(window=self.volatility_window).std()
                    
                    profile['pnl_volatility'] = rolling_pnl_vol.mean()
                    profile['pnl_volatility_trend'] = self._calculate_trend(rolling_pnl_vol.dropna().values)
        
        return profile
    
    def _calculate_single_user_volatility(self, user_row: pd.Series, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula volatilidade para dados agregados."""
        profile = {}
        
        # Usar desvio padrão como proxy de volatilidade se disponível
        vol_cols = ['bet_std', 'bet_volatility', 'std_bet_amount']
        avg_cols = ['avg_bet_amount', 'mean_bet_amount']
        
        vol_value = None
        avg_value = None
        
        for col in vol_cols:
            if col in user_row.index:
                vol_value = user_row.get(col, 0)
                break
        
        for col in avg_cols:
            if col in user_row.index:
                avg_value = user_row.get(col, 1)
                break
        
        if vol_value is not None and avg_value is not None and avg_value > 0:
            profile['normalized_volatility'] = vol_value / avg_value
        
        return profile
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calcula tendência em uma série temporal."""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _extract_profitability_features(self, user_data: Any, features_df: pd.DataFrame, 
                                      amount_cols: Dict[str, str]) -> pd.DataFrame:
        """Extrai features de rentabilidade."""
        logger.info("Extraindo features de rentabilidade...")
        
        profit_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                profit_features[user_id] = self._calculate_profitability(user_df, amount_cols)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    profit_features[idx] = self._calculate_single_user_profitability(user_row, amount_cols)
        
        # Converter para DataFrame
        profit_df = pd.DataFrame.from_dict(profit_features, orient='index')
        
        # Combinar com features_df
        for col in profit_df.columns:
            features_df[f'profit_{col}'] = profit_df[col]
        
        return features_df
    
    def _calculate_profitability(self, user_df: pd.DataFrame, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula métricas de rentabilidade."""
        profile = {}
        
        bet_col = amount_cols.get('bet', 'bet_amount')
        win_col = amount_cols.get('win', 'win_amount')
        
        if bet_col in user_df.columns and win_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            wins = user_df[win_col].fillna(0)
            
            if len(bets) > 0:
                total_wagered = bets.sum()
                total_won = wins.sum()
                
                # Métricas básicas de P&L
                net_result = total_won - total_wagered
                profile['total_pnl'] = net_result
                profile['roi'] = net_result / (total_wagered + 1e-8)
                profile['win_rate'] = (wins > 0).mean()
                
                # Return on Investment
                if total_wagered > 0:
                    profile['rtp'] = total_won / total_wagered  # Return to Player
                
                # Análise de streaks
                win_mask = wins > bets  # Vitórias líquidas
                profile['max_winning_streak'] = self._calculate_max_streak(win_mask, True)
                profile['max_losing_streak'] = self._calculate_max_streak(win_mask, False)
                
                # Eficiência dos ganhos
                winning_sessions = wins[wins > 0]
                if len(winning_sessions) > 0:
                    profile['avg_win_amount'] = winning_sessions.mean()
                    profile['max_win'] = winning_sessions.max()
                
                # Average win vs average loss
                losses = bets[wins == 0]
                if len(losses) > 0 and len(winning_sessions) > 0:
                    avg_loss = losses.mean()
                    avg_win = winning_sessions.mean()
                    profile['win_loss_ratio'] = avg_win / (avg_loss + 1e-8)
        
        return profile
    
    def _calculate_single_user_profitability(self, user_row: pd.Series, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula rentabilidade para dados agregados."""
        profile = {}
        
        # Win rate
        if 'win_rate' in user_row.index:
            profile['win_rate'] = user_row.get('win_rate', 0.5)
        
        # ROI se disponível
        if 'total_won' in user_row.index and 'total_wagered' in user_row.index:
            total_won = user_row.get('total_won', 0)
            total_wagered = user_row.get('total_wagered', 1)
            
            profile['roi'] = (total_won - total_wagered) / total_wagered
            profile['rtp'] = total_won / total_wagered
        
        return profile
    
    def _calculate_max_streak(self, mask: pd.Series, target_value: bool) -> int:
        """Calcula a maior sequência de um valor específico."""
        if len(mask) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for value in mask:
            if value == target_value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _extract_transaction_features(self, user_data: Any, features_df: pd.DataFrame, 
                                    amount_cols: Dict[str, str]) -> pd.DataFrame:
        """Extrai features de padrões transacionais."""
        logger.info("Extraindo features de padrões transacionais...")
        
        transaction_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                transaction_features[user_id] = self._calculate_transaction_patterns(user_df, amount_cols)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    transaction_features[idx] = self._calculate_single_user_transactions(user_row, amount_cols)
        
        # Converter para DataFrame
        transaction_df = pd.DataFrame.from_dict(transaction_features, orient='index')
        
        # Combinar com features_df
        for col in transaction_df.columns:
            features_df[f'transaction_{col}'] = transaction_df[col]
        
        return features_df
    
    def _calculate_transaction_patterns(self, user_df: pd.DataFrame, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula padrões transacionais."""
        profile = {}
        
        # Frequência transacional
        if 'timestamp' in user_df.columns:
            timestamps = pd.to_datetime(user_df['timestamp'])
            time_span = (timestamps.max() - timestamps.min()).total_seconds() / 86400  # dias
            
            if time_span > 0:
                profile['transaction_frequency'] = len(user_df) / time_span
        
        # Análise de volume por transação
        bet_col = amount_cols.get('bet', 'bet_amount')
        if bet_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            
            if len(bets) > 0:
                profile['transactions_count'] = len(bets)
                profile['avg_transaction_size'] = bets.mean()
                
                # Distribuição de tamanhos de transação
                small_bets = (bets < bets.quantile(0.33)).sum()
                medium_bets = ((bets >= bets.quantile(0.33)) & (bets < bets.quantile(0.67))).sum()
                large_bets = (bets >= bets.quantile(0.67)).sum()
                
                total_bets = len(bets)
                profile['small_bet_ratio'] = small_bets / total_bets
                profile['medium_bet_ratio'] = medium_bets / total_bets
                profile['large_bet_ratio'] = large_bets / total_bets
        
        return profile
    
    def _calculate_single_user_transactions(self, user_row: pd.Series, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula transações para dados agregados."""
        profile = {}
        
        # Contagem de transações
        if 'total_bets' in user_row.index:
            profile['transactions_count'] = user_row.get('total_bets', 0)
        
        # Frequência se temos informação temporal
        if 'session_frequency' in user_row.index:
            profile['transaction_frequency'] = user_row.get('session_frequency', 0)
        
        return profile
    
    def _extract_risk_management_features(self, user_data: Any, features_df: pd.DataFrame, 
                                        amount_cols: Dict[str, str]) -> pd.DataFrame:
        """Extrai features de gestão de risco."""
        logger.info("Extraindo features de gestão de risco...")
        
        risk_features = {}
        
        if isinstance(user_data, pd.core.groupby.generic.DataFrameGroupBy):
            for user_id, user_df in user_data:
                risk_features[user_id] = self._calculate_risk_management(user_df, amount_cols)
        else:
            for idx, user_row in user_data:
                if isinstance(user_row, pd.Series):
                    risk_features[idx] = self._calculate_single_user_risk_mgmt(user_row, amount_cols)
        
        # Converter para DataFrame
        risk_df = pd.DataFrame.from_dict(risk_features, orient='index')
        
        # Combinar com features_df
        for col in risk_df.columns:
            features_df[f'risk_mgmt_{col}'] = risk_df[col]
        
        return features_df
    
    def _calculate_risk_management(self, user_df: pd.DataFrame, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula gestão de risco."""
        profile = {}
        
        bet_col = amount_cols.get('bet', 'bet_amount')
        win_col = amount_cols.get('win', 'win_amount')
        
        if bet_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            
            if len(bets) >= self.loss_chase_window:
                # Detectar chase de perdas
                if win_col in user_df.columns:
                    wins = user_df[win_col].fillna(0)
                    
                    # Ordenar por timestamp se disponível
                    if 'timestamp' in user_df.columns:
                        user_df_sorted = user_df.sort_values('timestamp')
                        bets_sorted = user_df_sorted[bet_col].dropna()
                        wins_sorted = user_df_sorted[win_col].fillna(0)
                        
                        # Detectar padrão de chase
                        chase_count = 0
                        total_windows = len(bets_sorted) - self.loss_chase_window + 1
                        
                        for i in range(total_windows):
                            window_bets = bets_sorted.iloc[i:i+self.loss_chase_window]
                            window_wins = wins_sorted.iloc[i:i+self.loss_chase_window]
                            
                            # Se há perda seguida de aumento de aposta
                            if (window_wins.iloc[0] < window_bets.iloc[0] and  # Primeira é perda
                                window_bets.iloc[1] > window_bets.iloc[0] * 1.5):  # Segunda é 50% maior
                                chase_count += 1
                        
                        profile['loss_chase_frequency'] = chase_count / total_windows if total_windows > 0 else 0
                
                # Análise de stop-loss behavior
                if len(bets) > 5:
                    # Detectar sessões com declínio gradual vs stop abrupto
                    bet_changes = bets.pct_change().dropna()
                    large_decreases = (bet_changes < -0.5).sum()  # Reduções > 50%
                    
                    profile['stop_loss_discipline'] = large_decreases / len(bet_changes) if len(bet_changes) > 0 else 0
        
        # Value at Risk (VaR) simples
        if bet_col in user_df.columns:
            bets = user_df[bet_col].dropna()
            
            if len(bets) > 10:
                # VaR 95% = percentil 95 das apostas
                var_95 = bets.quantile(0.95)
                avg_bet = bets.mean()
                
                profile['var_95_ratio'] = var_95 / (avg_bet + 1e-8)
        
        return profile
    
    def _calculate_single_user_risk_mgmt(self, user_row: pd.Series, amount_cols: Dict[str, str]) -> Dict[str, float]:
        """Calcula gestão de risco para dados agregados."""
        profile = {}
        
        # Risk indicators baseados em ratios disponíveis
        if 'max_bet_amount' in user_row.index and 'avg_bet_amount' in user_row.index:
            max_bet = user_row.get('max_bet_amount', 0)
            avg_bet = user_row.get('avg_bet_amount', 1)
            
            # Ratio alto pode indicar poor risk management
            max_avg_ratio = max_bet / (avg_bet + 1e-8)
            profile['risk_concentration'] = min(max_avg_ratio / 10, 1)  # Normalizado
        
        return profile
    
    def get_financial_insights(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera insights das features financeiras extraídas.
        
        Args:
            features_df: DataFrame com features financeiras
            
        Returns:
            Dict com insights financeiros
        """
        insights = {}
        
        # Análise de segmentação por valor
        if 'spending_user_tier' in features_df.columns:
            tier_distribution = features_df['spending_user_tier'].value_counts()
            total_users = len(features_df)
            
            insights['user_segmentation'] = {
                'whales': tier_distribution.get(3, 0) / total_users,
                'high_rollers': tier_distribution.get(2, 0) / total_users,
                'regular_users': tier_distribution.get(1, 0) / total_users
            }
        
        # Análise de rentabilidade
        profit_cols = [col for col in features_df.columns if 'profit_roi' in col]
        if profit_cols:
            roi_col = profit_cols[0]
            roi_values = features_df[roi_col].dropna()
            
            if len(roi_values) > 0:
                insights['profitability'] = {
                    'profitable_users': (roi_values > 0).mean(),
                    'avg_roi': roi_values.mean(),
                    'median_roi': roi_values.median(),
                    'high_profit_users': (roi_values > 0.1).mean()  # > 10% ROI
                }
        
        # Análise de volatilidade
        volatility_cols = [col for col in features_df.columns if 'volatility_normalized' in col]
        if volatility_cols:
            vol_col = volatility_cols[0]
            vol_values = features_df[vol_col].dropna()
            
            if len(vol_values) > 0:
                insights['volatility'] = {
                    'low_volatility_users': (vol_values < 0.5).mean(),
                    'high_volatility_users': (vol_values > 2.0).mean(),
                    'avg_volatility': vol_values.mean()
                }
        
        return insights
    
    def export_financial_documentation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exporta documentação das features financeiras.
        
        Args:
            features_df: DataFrame com features
            
        Returns:
            pd.DataFrame: Documentação das features
        """
        documentation = []
        
        feature_descriptions = {
            # Spending features
            'spending_total_wagered': 'Valor total apostado pelo usuário',
            'spending_avg_bet': 'Valor médio das apostas',
            'spending_bet_cv': 'Coeficiente de variação das apostas',
            'spending_user_tier': 'Classificação do usuário (1=Regular, 2=High Roller, 3=Whale)',
            'spending_max_bet_ratio': 'Ratio entre aposta máxima e média',
            
            # Bankroll features
            'bankroll_total_deposits': 'Valor total de depósitos',
            'bankroll_bet_deposit_ratio': 'Ratio entre apostas e depósitos',
            'bankroll_sizing_discipline': 'Disciplina no sizing das apostas',
            
            # Volatility features
            'volatility_normalized_volatility': 'Volatilidade normalizada das apostas',
            'volatility_pnl_volatility': 'Volatilidade do P&L',
            
            # Profitability features
            'profit_roi': 'Return on Investment do usuário',
            'profit_win_rate': 'Taxa de vitórias',
            'profit_rtp': 'Return to Player (total ganho / total apostado)',
            'profit_max_winning_streak': 'Maior sequência de vitórias',
            
            # Transaction features
            'transaction_frequency': 'Frequência de transações por dia',
            'transaction_avg_transaction_size': 'Tamanho médio das transações',
            
            # Risk management features
            'risk_mgmt_loss_chase_frequency': 'Frequência de chase após perdas',
            'risk_mgmt_var_95_ratio': 'Value at Risk 95% normalizado'
        }
        
        for feature in features_df.columns:
            doc_entry = {
                'feature_name': feature,
                'category': feature.split('_')[0] if '_' in feature else 'financial',
                'description': feature_descriptions.get(feature, 'Feature financeira'),
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
    
    # Simular dados financeiros de usuários
    n_users = 400
    n_transactions_per_user = np.random.poisson(30, n_users)
    
    test_data = []
    
    for user_id in range(n_users):
        n_trans = n_transactions_per_user[user_id]
        
        # Diferentes perfis financeiros
        if user_id < 50:  # Whales
            base_bet = np.random.lognormal(7, 1)  # ~$1000+
            deposit_amounts = np.random.lognormal(8, 1, max(1, n_trans//10))
            win_rate = 0.48  # Slightly losing
        elif user_id < 150:  # High Rollers
            base_bet = np.random.lognormal(5, 1)  # ~$150
            deposit_amounts = np.random.lognormal(6, 1, max(1, n_trans//8))
            win_rate = 0.46
        elif user_id < 300:  # Regular players
            base_bet = np.random.lognormal(3, 0.8)  # ~$20
            deposit_amounts = np.random.lognormal(4, 0.8, max(1, n_trans//5))
            win_rate = 0.45
        else:  # Small players
            base_bet = np.random.lognormal(1, 0.5)  # ~$3
            deposit_amounts = np.random.lognormal(2, 0.5, max(1, n_trans//3))
            win_rate = 0.44
        
        # Gerar transações
        for trans_id in range(n_trans):
            # Variação na aposta
            bet_multiplier = np.random.lognormal(0, 0.3)  # Variação natural
            bet_amount = base_bet * bet_multiplier
            
            # Win amount baseado na win rate
            if np.random.random() < win_rate:
                # Ganhou - payout aleatório
                payout_multiplier = np.random.lognormal(0.5, 0.8)
                win_amount = bet_amount * payout_multiplier
            else:
                win_amount = 0
            
            # Timestamp aleatório
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 90))
            
            test_data.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'bet_amount': bet_amount,
                'win_amount': win_amount
            })
        
        # Adicionar depósitos
        for dep_id, deposit_amount in enumerate(deposit_amounts):
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 90))
            test_data.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'bet_amount': 0,
                'win_amount': 0,
                'deposit_amount': deposit_amount
            })
    
    test_df = pd.DataFrame(test_data)
    
    # Testar extração de features financeiras
    print("🚀 Testando Financial Feature Engine...")
    
    engine = FinancialFeatureEngine()
    financial_features = engine.extract_all_features(test_df)
    
    print(f"✅ Features financeiras extraídas: {len(financial_features.columns)}")
    print(f"✅ Usuários processados: {len(financial_features)}")
    
    # Mostrar algumas features
    print(f"\n✅ Primeiras features financeiras:")
    print(financial_features.head())
    
    # Insights financeiros
    insights = engine.get_financial_insights(financial_features)
    
    if 'user_segmentation' in insights:
        segmentation = insights['user_segmentation']
        print(f"\n✅ Segmentação de usuários:")
        print(f"   Whales: {segmentation['whales']:.1%}")
        print(f"   High Rollers: {segmentation['high_rollers']:.1%}")
        print(f"   Regular Users: {segmentation['regular_users']:.1%}")
    
    if 'profitability' in insights:
        profitability = insights['profitability']
        print(f"\n✅ Análise de rentabilidade:")
        print(f"   Usuários lucrativos: {profitability['profitable_users']:.1%}")
        print(f"   ROI médio: {profitability['avg_roi']:.2%}")
    
    # Documentação
    documentation = engine.export_financial_documentation(financial_features)
    print(f"\n✅ Documentação gerada para {len(documentation)} features financeiras")
    
    print("\n🎯 Financial Feature Engine implementado com sucesso!")