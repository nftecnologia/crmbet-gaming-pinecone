"""
Cluster Interpretation - Sistema de interpretação business-ready para clusters gaming/apostas
Implementação científica para gerar personas, estratégias e insights acionáveis.

@author: UltraThink Data Science Team  
@version: 1.0.0
@domain: Gaming/Apostas CRM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterInterpreter:
    """
    Sistema de interpretação de clusters para gaming/apostas.
    
    Features:
    - Geração automática de personas
    - Estratégias de campanha personalizadas
    - Identificação de características distintivas
    - Recomendações de ações CRM
    - Análise de valor e risco por cluster
    - Insights acionáveis para negócio
    """
    
    def __init__(self,
                 feature_categories: Dict[str, List[str]] = None,
                 business_priorities: List[str] = None,
                 campaign_templates: Dict[str, Dict] = None):
        """
        Inicializa o interpretador de clusters.
        
        Args:
            feature_categories: Categorização das features para interpretação
            business_priorities: Prioridades de negócio (revenue, retention, etc.)
            campaign_templates: Templates de campanhas por tipo de usuário
        """
        self.feature_categories = feature_categories or {
            'financial': ['spending_', 'financial_', 'profit_', 'bankroll_'],
            'behavioral': ['risk_', 'consistency_', 'engagement_'],
            'temporal': ['circadian_', 'weekly_', 'timing_'],
            'communication': ['channel_', 'campaign_', 'support_']
        }
        
        self.business_priorities = business_priorities or [
            'revenue_growth', 'retention', 'engagement', 'risk_management'
        ]
        
        self.campaign_templates = campaign_templates or self._default_campaign_templates()
        
        # Resultados da interpretação
        self.cluster_personas = {}
        self.cluster_strategies = {}
        self.actionable_insights = {}
        
    def _default_campaign_templates(self) -> Dict[str, Dict]:
        """Templates padrão de campanhas por tipo de cluster."""
        return {
            'high_value': {
                'name': 'VIP Experience',
                'channels': ['email', 'phone', 'in_app'],
                'frequency': 'weekly',
                'content_type': 'exclusive_offers',
                'personalization_level': 'high'
            },
            'at_risk': {
                'name': 'Retention Campaign',
                'channels': ['email', 'sms', 'push'],
                'frequency': 'bi_weekly',
                'content_type': 'winback_offers',
                'personalization_level': 'medium'
            },
            'casual': {
                'name': 'Engagement Boost',
                'channels': ['email', 'in_app'],
                'frequency': 'monthly',
                'content_type': 'educational_tips',
                'personalization_level': 'low'
            },
            'new_user': {
                'name': 'Welcome Journey',
                'channels': ['email', 'in_app', 'push'],
                'frequency': 'daily_first_week',
                'content_type': 'onboarding_tutorials',
                'personalization_level': 'medium'
            }
        }
    
    def interpret_clusters(self, 
                         cluster_profiles: Dict[int, Dict],
                         feature_matrix: pd.DataFrame,
                         cluster_labels: np.ndarray,
                         feature_importance: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Interpreta clusters completos gerando personas e estratégias.
        
        Args:
            cluster_profiles: Perfis estatísticos dos clusters
            feature_matrix: Matriz de features original
            cluster_labels: Labels dos clusters
            feature_importance: Importância das features
            
        Returns:
            Dict: Interpretação completa dos clusters
        """
        logger.info("Iniciando interpretação dos clusters...")
        
        interpretation = {
            'personas': {},
            'strategies': {},
            'insights': {},
            'recommendations': {},
            'metadata': {
                'interpreted_at': datetime.now().isoformat(),
                'n_clusters': len([k for k in cluster_profiles.keys() if k != -1]),
                'total_users': len(cluster_labels)
            }
        }
        
        # Gerar personas para cada cluster
        for cluster_id, profile in cluster_profiles.items():
            if cluster_id == -1:  # Tratar outliers separadamente
                continue
                
            logger.info(f"Interpretando cluster {cluster_id}...")
            
            # Gerar persona
            persona = self._generate_cluster_persona(
                cluster_id, profile, feature_matrix, cluster_labels, feature_importance
            )
            interpretation['personas'][cluster_id] = persona
            
            # Gerar estratégia
            strategy = self._generate_cluster_strategy(cluster_id, persona, profile)
            interpretation['strategies'][cluster_id] = strategy
        
        # Análise de outliers
        if -1 in cluster_profiles:
            outlier_analysis = self._analyze_outliers(
                cluster_profiles[-1], feature_matrix, cluster_labels
            )
            interpretation['outlier_analysis'] = outlier_analysis
        
        # Insights cross-cluster
        interpretation['insights'] = self._generate_cross_cluster_insights(
            interpretation['personas'], cluster_labels
        )
        
        # Recomendações estratégicas
        interpretation['recommendations'] = self._generate_strategic_recommendations(
            interpretation['personas'], interpretation['strategies']
        )
        
        # Salvar resultados
        self.cluster_personas = interpretation['personas']
        self.cluster_strategies = interpretation['strategies']
        self.actionable_insights = interpretation['insights']
        
        logger.info("Interpretação concluída com sucesso")
        return interpretation
    
    def _generate_cluster_persona(self, 
                                cluster_id: int,
                                profile: Dict,
                                feature_matrix: pd.DataFrame,
                                cluster_labels: np.ndarray,
                                feature_importance: Dict[str, float] = None) -> Dict[str, Any]:
        """Gera persona detalhada para um cluster."""
        
        cluster_mask = cluster_labels == cluster_id
        cluster_data = feature_matrix[cluster_mask]
        
        persona = {
            'cluster_id': cluster_id,
            'name': profile.get('name', f'Cluster_{cluster_id}'),
            'size': profile.get('size', 0),
            'percentage': profile.get('percentage', 0),
            'archetype': self._determine_archetype(profile),
            'characteristics': {},
            'behavioral_profile': {},
            'financial_profile': {},
            'communication_profile': {},
            'temporal_profile': {},
            'key_differentiators': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Analisar características por categoria
        statistics = profile.get('statistics', {})
        
        # Perfil financeiro
        persona['financial_profile'] = self._extract_financial_characteristics(statistics)
        
        # Perfil comportamental
        persona['behavioral_profile'] = self._extract_behavioral_characteristics(statistics)
        
        # Perfil de comunicação
        persona['communication_profile'] = self._extract_communication_characteristics(statistics)
        
        # Perfil temporal
        persona['temporal_profile'] = self._extract_temporal_characteristics(statistics)
        
        # Identificar diferenciadores chave
        persona['key_differentiators'] = self._identify_key_differentiators(
            cluster_data, feature_matrix, feature_importance
        )
        
        # Analisar riscos e oportunidades
        persona['risk_factors'] = self._identify_risk_factors(persona)
        persona['opportunities'] = self._identify_opportunities(persona)
        
        return persona
    
    def _determine_archetype(self, profile: Dict) -> str:
        """Determina o arquétipo do cluster baseado em características principais."""
        statistics = profile.get('statistics', {})
        
        # Extrair métricas chave
        avg_bet = self._safe_get_mean(statistics, 'spending_avg_bet', 0)
        frequency = self._safe_get_mean(statistics, 'engagement_frequency_score', 0)
        volatility = self._safe_get_mean(statistics, 'volatility_normalized_volatility', 0)
        response_rate = self._safe_get_mean(statistics, 'campaign_overall_response_rate', 0)
        
        # Lógica de classificação
        if avg_bet > 100:  # High value
            if volatility > 1.5:
                return "High Roller Risk Taker"
            else:
                return "VIP Conservative Player"
        elif avg_bet > 50:
            if frequency > 0.7:
                return "Regular Engaged Player"
            else:
                return "Moderate Occasional Player"
        elif frequency > 0.8:
            return "Frequent Small Bettor"
        elif response_rate > 0.3:
            return "Promotion-Responsive Player"
        elif volatility < 0.5:
            return "Cautious Conservative Player"
        else:
            return "Casual Gaming Enthusiast"
    
    def _safe_get_mean(self, statistics: Dict, key: str, default: float = 0) -> float:
        """Extrai valor médio de forma segura."""
        if key in statistics and isinstance(statistics[key], dict):
            return statistics[key].get('mean', default)
        return default
    
    def _extract_financial_characteristics(self, statistics: Dict) -> Dict[str, Any]:
        """Extrai características financeiras."""
        financial = {}
        
        # Ticket médio
        avg_bet = self._safe_get_mean(statistics, 'spending_avg_bet')
        if avg_bet > 0:
            if avg_bet >= 1000:
                financial['spending_tier'] = 'whale'
                financial['spending_description'] = 'Apostador de altíssimo valor'
            elif avg_bet >= 100:
                financial['spending_tier'] = 'high_roller'
                financial['spending_description'] = 'Apostador de alto valor'
            elif avg_bet >= 20:
                financial['spending_tier'] = 'regular'
                financial['spending_description'] = 'Apostador regular'
            else:
                financial['spending_tier'] = 'casual'
                financial['spending_description'] = 'Apostador casual'
        
        # Volatilidade
        volatility = self._safe_get_mean(statistics, 'volatility_normalized_volatility')
        if volatility > 2.0:
            financial['risk_profile'] = 'high_risk'
            financial['risk_description'] = 'Perfil de alto risco com apostas voláteis'
        elif volatility > 1.0:
            financial['risk_profile'] = 'moderate_risk'
            financial['risk_description'] = 'Perfil de risco moderado'
        else:
            financial['risk_profile'] = 'low_risk'
            financial['risk_description'] = 'Perfil conservador e consistente'
        
        # ROI
        roi = self._safe_get_mean(statistics, 'profit_roi')
        if roi > 0.1:
            financial['profitability'] = 'profitable'
        elif roi > -0.1:
            financial['profitability'] = 'break_even'
        else:
            financial['profitability'] = 'losing'
        
        return financial
    
    def _extract_behavioral_characteristics(self, statistics: Dict) -> Dict[str, Any]:
        """Extrai características comportamentais."""
        behavioral = {}
        
        # Consistência
        consistency = self._safe_get_mean(statistics, 'consistency_duration_consistency')
        if consistency > 0.8:
            behavioral['consistency_level'] = 'highly_consistent'
            behavioral['consistency_description'] = 'Comportamento muito previsível e regular'
        elif consistency > 0.5:
            behavioral['consistency_level'] = 'moderately_consistent'
            behavioral['consistency_description'] = 'Comportamento relativamente previsível'
        else:
            behavioral['consistency_level'] = 'inconsistent'
            behavioral['consistency_description'] = 'Comportamento volátil e imprevisível'
        
        # Engajamento
        engagement = self._safe_get_mean(statistics, 'engagement_session_intensity')
        if engagement > 0.8:
            behavioral['engagement_level'] = 'highly_engaged'
        elif engagement > 0.5:
            behavioral['engagement_level'] = 'moderately_engaged'
        else:
            behavioral['engagement_level'] = 'low_engaged'
        
        # Diversidade de jogos
        diversity = self._safe_get_mean(statistics, 'diversity_game_entropy')
        if diversity > 0.8:
            behavioral['game_preference'] = 'diverse_explorer'
        elif diversity > 0.5:
            behavioral['game_preference'] = 'moderate_variety'
        else:
            behavioral['game_preference'] = 'focused_specialist'
        
        return behavioral
    
    def _extract_communication_characteristics(self, statistics: Dict) -> Dict[str, Any]:
        """Extrai características de comunicação."""
        communication = {}
        
        # Canal preferido
        email_ratio = self._safe_get_mean(statistics, 'channel_email_usage_ratio')
        sms_ratio = self._safe_get_mean(statistics, 'channel_sms_usage_ratio')
        push_ratio = self._safe_get_mean(statistics, 'channel_push_usage_ratio')
        
        ratios = {'email': email_ratio, 'sms': sms_ratio, 'push': push_ratio}
        preferred_channel = max(ratios, key=ratios.get)
        
        communication['preferred_channel'] = preferred_channel
        communication['channel_diversity'] = self._safe_get_mean(statistics, 'channel_diversity')
        
        # Responsividade
        response_rate = self._safe_get_mean(statistics, 'campaign_overall_response_rate')
        if response_rate > 0.3:
            communication['responsiveness'] = 'highly_responsive'
            communication['responsiveness_description'] = 'Responde bem a campanhas'
        elif response_rate > 0.15:
            communication['responsiveness'] = 'moderately_responsive'
            communication['responsiveness_description'] = 'Responsividade média a campanhas'
        else:
            communication['responsiveness'] = 'low_responsive'
            communication['responsiveness_description'] = 'Baixa responsividade a campanhas'
        
        return communication
    
    def _extract_temporal_characteristics(self, statistics: Dict) -> Dict[str, Any]:
        """Extrai características temporais."""
        temporal = {}
        
        # Horário preferido
        preferred_hour = self._safe_get_mean(statistics, 'timing_preferred_contact_hour', 20)
        
        if 6 <= preferred_hour <= 11:
            temporal['activity_period'] = 'morning'
            temporal['activity_description'] = 'Ativo principalmente pela manhã'
        elif 12 <= preferred_hour <= 17:
            temporal['activity_period'] = 'afternoon'
            temporal['activity_description'] = 'Ativo principalmente à tarde'
        elif 18 <= preferred_hour <= 22:
            temporal['activity_period'] = 'evening'
            temporal['activity_description'] = 'Ativo principalmente à noite'
        else:
            temporal['activity_period'] = 'late_night'
            temporal['activity_description'] = 'Ativo principalmente madrugada/noite'
        
        # Padrão semanal
        weekend_ratio = self._safe_get_mean(statistics, 'weekly_weekend_activity_ratio')
        if weekend_ratio > 0.6:
            temporal['weekly_pattern'] = 'weekend_focused'
        elif weekend_ratio < 0.3:
            temporal['weekly_pattern'] = 'weekday_focused'
        else:
            temporal['weekly_pattern'] = 'balanced'
        
        return temporal
    
    def _identify_key_differentiators(self, 
                                   cluster_data: pd.DataFrame,
                                   all_data: pd.DataFrame,
                                   feature_importance: Dict[str, float] = None) -> List[str]:
        """Identifica características que mais diferenciam este cluster."""
        differentiators = []
        
        if feature_importance:
            # Usar features mais importantes
            top_features = list(feature_importance.keys())[:10]
        else:
            # Usar todas as features numéricas
            top_features = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in top_features[:5]:  # Top 5 diferenciadores
            if feature in cluster_data.columns:
                cluster_mean = cluster_data[feature].mean()
                global_mean = all_data[feature].mean()
                
                if abs(cluster_mean - global_mean) > all_data[feature].std():
                    if cluster_mean > global_mean:
                        direction = "significativamente maior"
                    else:
                        direction = "significativamente menor"
                    
                    feature_friendly = self._translate_feature_name(feature)
                    differentiators.append(f"{feature_friendly} {direction} que a média")
        
        return differentiators
    
    def _translate_feature_name(self, feature: str) -> str:
        """Traduz nome técnico da feature para descrição amigável."""
        translations = {
            'spending_avg_bet': 'Valor médio das apostas',
            'engagement_session_intensity': 'Intensidade das sessões',
            'volatility_normalized_volatility': 'Volatilidade das apostas',
            'campaign_overall_response_rate': 'Taxa de resposta a campanhas',
            'timing_preferred_contact_hour': 'Horário preferido de atividade',
            'financial_roi': 'Retorno sobre investimento',
            'consistency_duration_consistency': 'Consistência comportamental',
            'diversity_game_entropy': 'Diversidade de jogos'
        }
        
        return translations.get(feature, feature.replace('_', ' ').title())
    
    def _identify_risk_factors(self, persona: Dict) -> List[str]:
        """Identifica fatores de risco para o cluster."""
        risks = []
        
        financial = persona.get('financial_profile', {})
        behavioral = persona.get('behavioral_profile', {})
        communication = persona.get('communication_profile', {})
        
        # Riscos financeiros
        if financial.get('risk_profile') == 'high_risk':
            risks.append("Alto risco: apostas voláteis podem indicar problema de jogo")
        
        if financial.get('profitability') == 'losing':
            risks.append("Perdas consistentes podem levar ao abandono")
        
        # Riscos comportamentais
        if behavioral.get('consistency_level') == 'inconsistent':
            risks.append("Comportamento imprevisível dificulta retenção")
        
        if behavioral.get('engagement_level') == 'low_engaged':
            risks.append("Baixo engajamento aumenta risco de churn")
        
        # Riscos de comunicação
        if communication.get('responsiveness') == 'low_responsive':
            risks.append("Baixa responsividade limita eficácia das campanhas")
        
        return risks
    
    def _identify_opportunities(self, persona: Dict) -> List[str]:
        """Identifica oportunidades para o cluster."""
        opportunities = []
        
        financial = persona.get('financial_profile', {})
        behavioral = persona.get('behavioral_profile', {})
        communication = persona.get('communication_profile', {})
        
        # Oportunidades financeiras
        if financial.get('spending_tier') in ['whale', 'high_roller']:
            opportunities.append("Alto valor: potencial para programas VIP exclusivos")
        
        if financial.get('profitability') == 'profitable':
            opportunities.append("Usuário lucrativo: incentivar maior atividade")
        
        # Oportunidades comportamentais
        if behavioral.get('engagement_level') == 'highly_engaged':
            opportunities.append("Alto engajamento: potencial para cross-selling")
        
        if behavioral.get('game_preference') == 'diverse_explorer':
            opportunities.append("Explorador: receptivo a novos jogos e features")
        
        # Oportunidades de comunicação
        if communication.get('responsiveness') == 'highly_responsive':
            opportunities.append("Altamente responsivo: ideal para campanhas direcionadas")
        
        return opportunities
    
    def _generate_cluster_strategy(self, 
                                 cluster_id: int,
                                 persona: Dict,
                                 profile: Dict) -> Dict[str, Any]:
        """Gera estratégia específica para o cluster."""
        
        financial_profile = persona.get('financial_profile', {})
        behavioral_profile = persona.get('behavioral_profile', {})
        communication_profile = persona.get('communication_profile', {})
        temporal_profile = persona.get('temporal_profile', {})
        
        strategy = {
            'cluster_id': cluster_id,
            'primary_objective': self._determine_primary_objective(persona),
            'campaign_strategy': self._design_campaign_strategy(persona),
            'communication_plan': self._design_communication_plan(persona),
            'retention_strategy': self._design_retention_strategy(persona),
            'growth_opportunities': self._identify_growth_opportunities(persona),
            'risk_mitigation': self._design_risk_mitigation(persona),
            'success_metrics': self._define_success_metrics(persona)
        }
        
        return strategy
    
    def _determine_primary_objective(self, persona: Dict) -> str:
        """Determina objetivo primário para o cluster."""
        financial = persona.get('financial_profile', {})
        behavioral = persona.get('behavioral_profile', {})
        risks = persona.get('risk_factors', [])
        
        if financial.get('spending_tier') in ['whale', 'high_roller']:
            return "retention_and_growth"
        elif len(risks) > 2:
            return "risk_mitigation"
        elif behavioral.get('engagement_level') == 'low_engaged':
            return "engagement_boost"
        else:
            return "value_optimization"
    
    def _design_campaign_strategy(self, persona: Dict) -> Dict[str, Any]:
        """Desenha estratégia de campanha para o cluster."""
        financial = persona.get('financial_profile', {})
        communication = persona.get('communication_profile', {})
        
        # Selecionar template base
        if financial.get('spending_tier') in ['whale', 'high_roller']:
            base_template = self.campaign_templates['high_value']
        elif len(persona.get('risk_factors', [])) > 2:
            base_template = self.campaign_templates['at_risk']
        else:
            base_template = self.campaign_templates['casual']
        
        # Customizar baseado no persona
        strategy = base_template.copy()
        strategy['preferred_channel'] = communication.get('preferred_channel', 'email')
        strategy['expected_response_rate'] = communication.get('responsiveness', 'moderate')
        
        return strategy
    
    def _design_communication_plan(self, persona: Dict) -> Dict[str, Any]:
        """Desenha plano de comunicação específico."""
        communication = persona.get('communication_profile', {})
        temporal = persona.get('temporal_profile', {})
        
        plan = {
            'primary_channel': communication.get('preferred_channel', 'email'),
            'backup_channels': ['sms', 'push'],  # Baseado na responsividade
            'optimal_timing': temporal.get('activity_period', 'evening'),
            'frequency': self._determine_communication_frequency(persona),
            'personalization_level': communication.get('responsiveness', 'moderate'),
            'content_preferences': self._determine_content_preferences(persona)
        }
        
        return plan
    
    def _determine_communication_frequency(self, persona: Dict) -> str:
        """Determina frequência ideal de comunicação."""
        engagement = persona.get('behavioral_profile', {}).get('engagement_level', 'moderate')
        responsiveness = persona.get('communication_profile', {}).get('responsiveness', 'moderate')
        
        if engagement == 'highly_engaged' and responsiveness == 'highly_responsive':
            return 'weekly'
        elif engagement == 'low_engaged' or responsiveness == 'low_responsive':
            return 'monthly'
        else:
            return 'bi_weekly'
    
    def _determine_content_preferences(self, persona: Dict) -> List[str]:
        """Determina preferências de conteúdo."""
        financial = persona.get('financial_profile', {})
        behavioral = persona.get('behavioral_profile', {})
        
        preferences = []
        
        if financial.get('spending_tier') in ['whale', 'high_roller']:
            preferences.extend(['exclusive_offers', 'vip_events', 'personal_account_manager'])
        
        if behavioral.get('game_preference') == 'diverse_explorer':
            preferences.extend(['new_game_announcements', 'game_tutorials'])
        
        if financial.get('risk_profile') == 'high_risk':
            preferences.extend(['responsible_gaming_tips', 'limit_setting_tools'])
        
        if not preferences:
            preferences = ['general_promotions', 'game_tips', 'community_updates']
        
        return preferences
    
    def _design_retention_strategy(self, persona: Dict) -> Dict[str, Any]:
        """Desenha estratégia de retenção."""
        risks = persona.get('risk_factors', [])
        financial = persona.get('financial_profile', {})
        behavioral = persona.get('behavioral_profile', {})
        
        strategy = {
            'risk_level': 'low' if len(risks) < 2 else 'high',
            'retention_tactics': [],
            'early_warning_indicators': [],
            'intervention_triggers': []
        }
        
        # Táticas baseadas no perfil
        if financial.get('spending_tier') in ['whale', 'high_roller']:
            strategy['retention_tactics'].extend([
                'vip_treatment', 'dedicated_support', 'exclusive_bonuses'
            ])
        
        if behavioral.get('engagement_level') == 'low_engaged':
            strategy['retention_tactics'].extend([
                'engagement_campaigns', 'educational_content', 'gamification'
            ])
        
        # Indicadores de alerta
        strategy['early_warning_indicators'] = [
            'decreased_session_frequency', 'reduced_bet_amounts', 'communication_opt_outs'
        ]
        
        return strategy
    
    def _identify_growth_opportunities(self, persona: Dict) -> List[str]:
        """Identifica oportunidades de crescimento."""
        opportunities = persona.get('opportunities', [])
        financial = persona.get('financial_profile', {})
        behavioral = persona.get('behavioral_profile', {})
        
        growth_ops = []
        
        if 'cross-selling' in ' '.join(opportunities).lower():
            growth_ops.append("Cross-selling de novos produtos de jogo")
        
        if financial.get('profitability') == 'profitable':
            growth_ops.append("Upselling para apostas de maior valor")
        
        if behavioral.get('game_preference') == 'diverse_explorer':
            growth_ops.append("Introdução a novos tipos de jogos")
        
        return growth_ops
    
    def _design_risk_mitigation(self, persona: Dict) -> Dict[str, Any]:
        """Desenha estratégia de mitigação de riscos."""
        risks = persona.get('risk_factors', [])
        
        mitigation = {
            'priority_level': 'high' if len(risks) > 2 else 'medium',
            'specific_actions': [],
            'monitoring_metrics': [],
            'escalation_procedures': []
        }
        
        for risk in risks:
            if 'problema de jogo' in risk.lower():
                mitigation['specific_actions'].append('Implementar limites de jogo responsável')
                mitigation['monitoring_metrics'].append('Frequency of limit breaches')
            
            elif 'abandono' in risk.lower():
                mitigation['specific_actions'].append('Campanhas de retenção proativas')
                mitigation['monitoring_metrics'].append('Days since last activity')
            
            elif 'churn' in risk.lower():
                mitigation['specific_actions'].append('Programa de engajamento intensivo')
                mitigation['monitoring_metrics'].append('Engagement score decline')
        
        return mitigation
    
    def _define_success_metrics(self, persona: Dict) -> Dict[str, str]:
        """Define métricas de sucesso para o cluster."""
        financial = persona.get('financial_profile', {})
        
        metrics = {
            'primary_kpi': 'customer_lifetime_value',
            'secondary_kpis': ['retention_rate', 'engagement_score'],
            'business_metrics': ['revenue_per_user', 'session_frequency'],
            'risk_metrics': ['responsible_gaming_score', 'complaint_rate']
        }
        
        # Customizar baseado no perfil
        if financial.get('spending_tier') in ['whale', 'high_roller']:
            metrics['primary_kpi'] = 'revenue_retention'
        elif len(persona.get('risk_factors', [])) > 2:
            metrics['primary_kpi'] = 'risk_mitigation_success'
        
        return metrics
    
    def _analyze_outliers(self, 
                        outlier_profile: Dict,
                        feature_matrix: pd.DataFrame,
                        cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analisa cluster de outliers."""
        outlier_mask = cluster_labels == -1
        outlier_data = feature_matrix[outlier_mask]
        
        analysis = {
            'count': outlier_profile.get('size', 0),
            'percentage': outlier_profile.get('percentage', 0),
            'characteristics': 'Usuários com comportamento anômalo',
            'recommended_actions': [
                'Análise individual caso a caso',
                'Verificação de fraude ou abuso',
                'Potencial para segmento personalizado',
                'Monitoramento especializado'
            ],
            'risk_assessment': 'Alto - requer atenção especial'
        }
        
        return analysis
    
    def _generate_cross_cluster_insights(self, 
                                       personas: Dict[int, Dict],
                                       cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Gera insights que atravessam múltiplos clusters."""
        insights = {
            'cluster_distribution': self._analyze_cluster_distribution(personas, cluster_labels),
            'value_concentration': self._analyze_value_concentration(personas),
            'risk_distribution': self._analyze_risk_distribution(personas),
            'communication_patterns': self._analyze_communication_patterns(personas),
            'temporal_patterns': self._analyze_temporal_patterns(personas)
        }
        
        return insights
    
    def _analyze_cluster_distribution(self, personas: Dict, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analisa distribuição dos clusters."""
        total_users = len(cluster_labels)
        
        distribution = {
            'total_clusters': len(personas),
            'largest_cluster': None,
            'smallest_cluster': None,
            'balance_score': 0
        }
        
        sizes = []
        for cluster_id, persona in personas.items():
            size = persona.get('size', 0)
            sizes.append(size)
            
            if distribution['largest_cluster'] is None or size > personas[distribution['largest_cluster']]['size']:
                distribution['largest_cluster'] = cluster_id
            
            if distribution['smallest_cluster'] is None or size < personas[distribution['smallest_cluster']]['size']:
                distribution['smallest_cluster'] = cluster_id
        
        # Balance score (menor variação = melhor balance)
        if sizes:
            distribution['balance_score'] = 1 - (np.std(sizes) / np.mean(sizes))
        
        return distribution
    
    def _analyze_value_concentration(self, personas: Dict) -> Dict[str, Any]:
        """Analisa concentração de valor entre clusters."""
        high_value_clusters = []
        total_users = sum(persona.get('size', 0) for persona in personas.values())
        
        for cluster_id, persona in personas.items():
            financial = persona.get('financial_profile', {})
            if financial.get('spending_tier') in ['whale', 'high_roller']:
                high_value_clusters.append({
                    'cluster_id': cluster_id,
                    'name': persona.get('name', f'Cluster_{cluster_id}'),
                    'size': persona.get('size', 0),
                    'tier': financial.get('spending_tier')
                })
        
        high_value_users = sum(cluster['size'] for cluster in high_value_clusters)
        
        return {
            'high_value_clusters': high_value_clusters,
            'high_value_concentration': high_value_users / total_users if total_users > 0 else 0,
            'pareto_analysis': "20/80 rule" if high_value_users / total_users < 0.2 else "Distributed value"
        }
    
    def _analyze_risk_distribution(self, personas: Dict) -> Dict[str, Any]:
        """Analisa distribuição de risco entre clusters."""
        risk_analysis = {
            'high_risk_clusters': [],
            'low_risk_clusters': [],
            'total_risk_score': 0
        }
        
        total_users = sum(persona.get('size', 0) for persona in personas.values())
        risk_weighted_sum = 0
        
        for cluster_id, persona in personas.items():
            risks = persona.get('risk_factors', [])
            risk_score = len(risks) / 5  # Normalizar por máximo de 5 riscos
            
            risk_weighted_sum += risk_score * persona.get('size', 0)
            
            if len(risks) > 2:
                risk_analysis['high_risk_clusters'].append({
                    'cluster_id': cluster_id,
                    'name': persona.get('name'),
                    'risk_count': len(risks)
                })
            elif len(risks) < 1:
                risk_analysis['low_risk_clusters'].append({
                    'cluster_id': cluster_id,
                    'name': persona.get('name'),
                    'risk_count': len(risks)
                })
        
        risk_analysis['total_risk_score'] = risk_weighted_sum / total_users if total_users > 0 else 0
        
        return risk_analysis
    
    def _analyze_communication_patterns(self, personas: Dict) -> Dict[str, Any]:
        """Analisa padrões de comunicação entre clusters."""
        channel_preferences = {}
        responsiveness_levels = {'highly_responsive': 0, 'moderately_responsive': 0, 'low_responsive': 0}
        
        for persona in personas.values():
            comm = persona.get('communication_profile', {})
            
            # Canais preferenciais
            preferred = comm.get('preferred_channel', 'email')
            channel_preferences[preferred] = channel_preferences.get(preferred, 0) + persona.get('size', 0)
            
            # Responsividade
            responsiveness = comm.get('responsiveness', 'moderately_responsive')
            responsiveness_levels[responsiveness] += persona.get('size', 0)
        
        return {
            'dominant_channels': dict(sorted(channel_preferences.items(), key=lambda x: x[1], reverse=True)),
            'responsiveness_distribution': responsiveness_levels
        }
    
    def _analyze_temporal_patterns(self, personas: Dict) -> Dict[str, Any]:
        """Analisa padrões temporais entre clusters."""
        activity_periods = {}
        weekly_patterns = {}
        
        for persona in personas.values():
            temporal = persona.get('temporal_profile', {})
            
            # Períodos de atividade
            period = temporal.get('activity_period', 'evening')
            activity_periods[period] = activity_periods.get(period, 0) + persona.get('size', 0)
            
            # Padrões semanais
            pattern = temporal.get('weekly_pattern', 'balanced')
            weekly_patterns[pattern] = weekly_patterns.get(pattern, 0) + persona.get('size', 0)
        
        return {
            'peak_activity_periods': dict(sorted(activity_periods.items(), key=lambda x: x[1], reverse=True)),
            'weekly_behavior_patterns': weekly_patterns
        }
    
    def _generate_strategic_recommendations(self, 
                                          personas: Dict,
                                          strategies: Dict) -> Dict[str, Any]:
        """Gera recomendações estratégicas de alto nível."""
        recommendations = {
            'priority_clusters': [],
            'resource_allocation': {},
            'campaign_optimization': [],
            'risk_management': [],
            'growth_initiatives': []
        }
        
        # Identificar clusters prioritários
        for cluster_id, persona in personas.items():
            financial = persona.get('financial_profile', {})
            risks = persona.get('risk_factors', [])
            
            if financial.get('spending_tier') in ['whale', 'high_roller']:
                recommendations['priority_clusters'].append({
                    'cluster_id': cluster_id,
                    'name': persona.get('name'),
                    'reason': 'High value customer segment',
                    'priority_level': 'high'
                })
            elif len(risks) > 2:
                recommendations['priority_clusters'].append({
                    'cluster_id': cluster_id,
                    'name': persona.get('name'),
                    'reason': 'High risk requiring immediate attention',
                    'priority_level': 'urgent'
                })
        
        # Recomendações de campanha
        recommendations['campaign_optimization'] = [
            "Personalizar campanhas por cluster para aumentar efetividade",
            "Focar em canais preferenciais de cada segmento",
            "Ajustar frequência baseado na responsividade do cluster",
            "Desenvolver conteúdo específico para cada persona"
        ]
        
        # Gestão de risco
        recommendations['risk_management'] = [
            "Implementar monitoramento proativo para clusters de alto risco",
            "Desenvolver programas de jogo responsável segmentados",
            "Criar alertas automáticos para mudanças comportamentais",
            "Treinar equipe de suporte para características de cada cluster"
        ]
        
        return recommendations
    
    def export_business_summary(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Exporta resumo executivo para apresentação ao negócio.
        
        Args:
            save_path: Caminho para salvar JSON
            
        Returns:
            Dict: Resumo executivo
        """
        if not self.cluster_personas:
            raise ValueError("Nenhuma interpretação disponível. Execute interpret_clusters() primeiro.")
        
        summary = {
            'executive_summary': {
                'total_clusters': len(self.cluster_personas),
                'key_segments': [],
                'business_impact': {},
                'strategic_priorities': []
            },
            'segment_overview': {},
            'action_plan': {
                'immediate_actions': [],
                'short_term_goals': [],
                'long_term_strategy': []
            },
            'success_metrics': {
                'kpis_to_track': [],
                'expected_outcomes': []
            }
        }
        
        # Resumo executivo
        high_value_count = 0
        at_risk_count = 0
        
        for cluster_id, persona in self.cluster_personas.items():
            financial = persona.get('financial_profile', {})
            risks = persona.get('risk_factors', [])
            
            if financial.get('spending_tier') in ['whale', 'high_roller']:
                high_value_count += 1
                summary['executive_summary']['key_segments'].append({
                    'name': persona.get('name'),
                    'type': 'high_value',
                    'size': persona.get('size'),
                    'description': persona.get('archetype')
                })
            
            if len(risks) > 2:
                at_risk_count += 1
        
        summary['executive_summary']['business_impact'] = {
            'high_value_segments': high_value_count,
            'at_risk_segments': at_risk_count,
            'total_addressable_segments': len(self.cluster_personas)
        }
        
        # Plano de ação
        summary['action_plan']['immediate_actions'] = [
            "Implementar campanhas personalizadas por cluster",
            "Estabelecer monitoramento de clusters de alto risco",
            "Criar programas VIP para segmentos de alto valor"
        ]
        
        summary['action_plan']['short_term_goals'] = [
            "Aumentar responsividade de campanhas em 25%",
            "Reduzir churn em clusters de risco em 15%",
            "Incrementar CLV de clusters de alto valor em 20%"
        ]
        
        # Salvar se solicitado
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Resumo executivo salvo em: {save_path}")
        
        return summary


# Exemplo de uso e teste
if __name__ == "__main__":
    # Dados de exemplo para teste
    np.random.seed(42)
    
    # Simular cluster profiles
    cluster_profiles = {
        0: {
            'name': 'High Roller Enthusiasts',
            'size': 50,
            'percentage': 10.0,
            'statistics': {
                'spending_avg_bet': {'mean': 500.0, 'std': 200.0},
                'volatility_normalized_volatility': {'mean': 2.5, 'std': 0.8},
                'engagement_session_intensity': {'mean': 0.9, 'std': 0.1},
                'campaign_overall_response_rate': {'mean': 0.4, 'std': 0.1},
                'channel_email_usage_ratio': {'mean': 0.7, 'std': 0.2},
                'timing_preferred_contact_hour': {'mean': 22.0, 'std': 2.0}
            }
        },
        1: {
            'name': 'Casual Weekend Players',
            'size': 200,
            'percentage': 40.0,
            'statistics': {
                'spending_avg_bet': {'mean': 25.0, 'std': 10.0},
                'volatility_normalized_volatility': {'mean': 0.8, 'std': 0.3},
                'engagement_session_intensity': {'mean': 0.5, 'std': 0.2},
                'campaign_overall_response_rate': {'mean': 0.15, 'std': 0.05},
                'channel_sms_usage_ratio': {'mean': 0.6, 'std': 0.2},
                'timing_preferred_contact_hour': {'mean': 19.0, 'std': 3.0}
            }
        },
        2: {
            'name': 'At-Risk Declining Users',
            'size': 80,
            'percentage': 16.0,
            'statistics': {
                'spending_avg_bet': {'mean': 15.0, 'std': 8.0},
                'volatility_normalized_volatility': {'mean': 0.3, 'std': 0.1},
                'engagement_session_intensity': {'mean': 0.2, 'std': 0.1},
                'campaign_overall_response_rate': {'mean': 0.05, 'std': 0.02},
                'channel_email_usage_ratio': {'mean': 0.8, 'std': 0.1},
                'timing_preferred_contact_hour': {'mean': 14.0, 'std': 4.0}
            }
        }
    }
    
    # Simular feature matrix e labels
    n_total = sum(profile['size'] for profile in cluster_profiles.values())
    feature_matrix = pd.DataFrame(np.random.randn(n_total, 10), 
                                columns=[f'feature_{i}' for i in range(10)])
    
    cluster_labels = []
    for cluster_id, profile in cluster_profiles.items():
        cluster_labels.extend([cluster_id] * profile['size'])
    cluster_labels = np.array(cluster_labels)
    
    # Feature importance simulada
    feature_importance = {
        'spending_avg_bet': 0.25,
        'engagement_session_intensity': 0.20,
        'volatility_normalized_volatility': 0.15,
        'campaign_overall_response_rate': 0.12,
        'timing_preferred_contact_hour': 0.10
    }
    
    # Testar interpretador
    print("🚀 Testando Cluster Interpreter...")
    
    interpreter = ClusterInterpreter()
    
    # Interpretar clusters
    interpretation = interpreter.interpret_clusters(
        cluster_profiles, feature_matrix, cluster_labels, feature_importance
    )
    
    print(f"\n✅ Interpretação concluída:")
    print(f"   Clusters interpretados: {len(interpretation['personas'])}")
    print(f"   Estratégias geradas: {len(interpretation['strategies'])}")
    
    # Mostrar personas
    print(f"\n✅ Personas geradas:")
    for cluster_id, persona in interpretation['personas'].items():
        print(f"\n   Cluster {cluster_id}: {persona['name']}")
        print(f"   Arquétipo: {persona['archetype']}")
        print(f"   Tamanho: {persona['size']} usuários ({persona['percentage']:.1f}%)")
        
        financial = persona.get('financial_profile', {})
        print(f"   Perfil financeiro: {financial.get('spending_tier', 'N/A')}")
        
        if persona.get('key_differentiators'):
            print(f"   Diferenciadores: {persona['key_differentiators'][0]}")
        
        if persona.get('opportunities'):
            print(f"   Principal oportunidade: {persona['opportunities'][0]}")
    
    # Mostrar estratégias
    print(f"\n✅ Estratégias principais:")
    for cluster_id, strategy in interpretation['strategies'].items():
        persona_name = interpretation['personas'][cluster_id]['name']
        print(f"\n   {persona_name}:")
        print(f"   Objetivo: {strategy['primary_objective']}")
        
        campaign = strategy.get('campaign_strategy', {})
        print(f"   Campanha: {campaign.get('name', 'N/A')}")
        print(f"   Canal preferido: {campaign.get('preferred_channel', 'N/A')}")
    
    # Exportar resumo executivo
    business_summary = interpreter.export_business_summary()
    
    print(f"\n✅ Resumo executivo:")
    exec_summary = business_summary['executive_summary']
    print(f"   Total de segmentos: {exec_summary['total_clusters']}")
    print(f"   Segmentos de alto valor: {exec_summary['business_impact']['high_value_segments']}")
    print(f"   Segmentos em risco: {exec_summary['business_impact']['at_risk_segments']}")
    
    print(f"\n✅ Ações imediatas recomendadas:")
    for action in business_summary['action_plan']['immediate_actions'][:2]:
        print(f"   - {action}")
    
    print("\n🎯 Cluster Interpreter implementado com sucesso!")