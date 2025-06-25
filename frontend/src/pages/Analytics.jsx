import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  PieChart, 
  Calendar,
  Download,
  Filter,
  RefreshCw,
  Users,
  Mail,
  Target,
  DollarSign,
  Activity,
  Clock,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react'
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2'

const Analytics = () => {
  const [timeRange, setTimeRange] = useState('30d')
  const [selectedMetric, setSelectedMetric] = useState('all')

  // Mock data for analytics
  const metrics = [
    {
      title: 'Receita Total',
      value: 'R$ 284.5k',
      change: '+12.5%',
      trend: 'up',
      icon: DollarSign,
      color: 'success',
      description: 'vs período anterior'
    },
    {
      title: 'Conversão Média',
      value: '6.8%',
      change: '+0.8%',
      trend: 'up',
      icon: Target,
      color: 'primary',
      description: 'taxa de conversão geral'
    },
    {
      title: 'Usuários Ativos',
      value: '18.2k',
      change: '-2.3%',
      trend: 'down',
      icon: Users,
      color: 'warning',
      description: 'usuários únicos ativos'
    },
    {
      title: 'CAC Médio',
      value: 'R$ 23.50',
      change: '-8.2%',
      trend: 'up',
      icon: TrendingUp,
      color: 'info',
      description: 'custo de aquisição'
    }
  ]

  // Revenue trend chart
  const revenueData = {
    labels: ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    datasets: [
      {
        label: 'Receita',
        data: [18500, 22000, 28000, 32000, 25000, 31000, 38000, 35000, 42000, 45000, 38000, 48000],
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Meta',
        data: [20000, 23000, 26000, 29000, 32000, 35000, 38000, 41000, 44000, 47000, 50000, 53000],
        borderColor: '#94a3b8',
        backgroundColor: 'transparent',
        borderDash: [5, 5],
        fill: false,
        tension: 0.4,
      }
    ]
  }

  // Cluster performance chart
  const clusterData = {
    labels: ['Alto Valor', 'Médio Valor', 'Baixo Valor', 'Novo Cliente', 'VIP Premium', 'Inativo'],
    datasets: [
      {
        label: 'Receita (R$ mil)',
        data: [125, 85, 45, 32, 78, 5],
        backgroundColor: '#3b82f6',
        borderRadius: 6,
      },
      {
        label: 'Usuários (mil)',
        data: [3.2, 5.7, 8.9, 2.1, 0.45, 1.9],
        backgroundColor: '#22c55e',
        borderRadius: 6,
      }
    ]
  }

  // Conversion funnel data
  const funnelData = {
    labels: ['Visitantes', 'Leads', 'Oportunidades', 'Clientes', 'Recompra'],
    datasets: [
      {
        data: [45000, 12500, 3800, 1250, 450],
        backgroundColor: [
          '#3b82f6',
          '#22c55e',
          '#f59e0b',
          '#ef4444',
          '#8b5cf6'
        ],
        borderWidth: 0,
      }
    ]
  }

  // Heatmap data (simplified as radar for demo)
  const heatmapData = {
    labels: ['00-04h', '04-08h', '08-12h', '12-16h', '16-20h', '20-24h'],
    datasets: [
      {
        label: 'Segunda',
        data: [20, 35, 85, 92, 88, 65],
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: '#3b82f6',
        pointBackgroundColor: '#3b82f6',
      },
      {
        label: 'Sexta',
        data: [15, 25, 75, 88, 95, 78],
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderColor: '#22c55e',
        pointBackgroundColor: '#22c55e',
      }
    ]
  }

  const campaignPerformance = [
    {
      name: 'Black Friday 2024',
      cluster: 'Alto Valor',
      sent: 2450,
      opened: 1673,
      clicked: 412,
      converted: 103,
      revenue: 15450,
      roi: 285,
      openRate: 68.3,
      clickRate: 24.6,
      conversionRate: 4.2
    },
    {
      name: 'Boas-vindas Novos',
      cluster: 'Novo Cliente',
      sent: 2150,
      opened: 1469,
      clicked: 588,
      converted: 275,
      revenue: 8450,
      roi: 156,
      openRate: 68.3,
      clickRate: 40.0,
      conversionRate: 12.8
    },
    {
      name: 'Reativação',
      cluster: 'Inativo',
      sent: 1890,
      opened: 728,
      clicked: 146,
      converted: 32,
      revenue: 2890,
      roi: 89,
      openRate: 38.5,
      clickRate: 20.1,
      conversionRate: 2.2
    }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900">Analytics & Insights</h1>
          <p className="mt-1 text-sm text-secondary-600">
            Análise avançada de performance e insights de Machine Learning
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex items-center space-x-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="form-input"
          >
            <option value="7d">Últimos 7 dias</option>
            <option value="30d">Últimos 30 dias</option>
            <option value="90d">Últimos 90 dias</option>
            <option value="12m">Últimos 12 meses</option>
          </select>
          <button className="btn-secondary btn-md">
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </button>
          <button className="btn-secondary btn-md">
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="metric-card group"
          >
            <div className="flex items-center">
              <div className={`p-3 rounded-lg bg-${metric.color}-100 group-hover:bg-${metric.color}-200 transition-colors`}>
                <metric.icon className={`h-6 w-6 text-${metric.color}-600`} />
              </div>
              <div className="ml-4 flex-1">
                <p className="metric-label">{metric.title}</p>
                <div className="flex items-center space-x-2">
                  <p className="metric-value">{metric.value}</p>
                  <div className={`metric-change ${metric.trend === 'up' ? 'positive' : 'negative'}`}>
                    {metric.trend === 'up' ? (
                      <ArrowUpRight className="h-4 w-4" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4" />
                    )}
                    <span>{metric.change}</span>
                  </div>
                </div>
                <p className="text-xs text-secondary-500 mt-1">{metric.description}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Revenue Trend */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 card"
        >
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-secondary-900">
                Receita vs Meta - 2024
              </h3>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-3 h-3 bg-success-500 rounded-full"></div>
                  <span className="text-secondary-600">Receita</span>
                </div>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-3 h-3 border-2 border-dashed border-secondary-400 rounded-full"></div>
                  <span className="text-secondary-600">Meta</span>
                </div>
              </div>
            </div>
          </div>
          <div className="card-body">
            <div className="h-80">
              <Line 
                data={revenueData} 
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: false,
                    },
                  },
                  scales: {
                    x: {
                      grid: {
                        display: false,
                      },
                    },
                    y: {
                      grid: {
                        color: '#f1f5f9',
                      },
                      ticks: {
                        callback: function(value) {
                          return 'R$ ' + (value / 1000) + 'k'
                        }
                      }
                    },
                  },
                }} 
              />
            </div>
          </div>
        </motion.div>

        {/* Conversion Funnel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Funil de Conversão
            </h3>
          </div>
          <div className="card-body">
            <div className="h-60">
              <Doughnut 
                data={funnelData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                          size: 11,
                        },
                      },
                    },
                  },
                }}
              />
            </div>
          </div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cluster Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Performance por Cluster
            </h3>
          </div>
          <div className="card-body">
            <div className="h-60">
              <Bar 
                data={clusterData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'top',
                    },
                  },
                  scales: {
                    x: {
                      grid: {
                        display: false,
                      },
                    },
                    y: {
                      grid: {
                        color: '#f1f5f9',
                      },
                    },
                  },
                }}
              />
            </div>
          </div>
        </motion.div>

        {/* Activity Heatmap */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Padrão de Atividade
            </h3>
          </div>
          <div className="card-body">
            <div className="h-60">
              <Radar 
                data={heatmapData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'top',
                    },
                  },
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 100,
                    },
                  },
                }}
              />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Campaign Performance Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="card"
      >
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-secondary-900">
              Performance de Campanhas
            </h3>
            <button className="btn-ghost btn-sm">
              <Filter className="h-4 w-4 mr-2" />
              Filtrar
            </button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Campanha</th>
                <th>Cluster</th>
                <th>Enviados</th>
                <th>Taxa Abertura</th>
                <th>Taxa Clique</th>
                <th>Conversão</th>
                <th>Receita</th>
                <th>ROI</th>
              </tr>
            </thead>
            <tbody>
              {campaignPerformance.map((campaign) => (
                <tr key={campaign.name}>
                  <td>
                    <div className="font-medium text-secondary-900">{campaign.name}</div>
                  </td>
                  <td>
                    <span className="badge badge-info">{campaign.cluster}</span>
                  </td>
                  <td>
                    <div className="text-sm text-secondary-900">{campaign.sent.toLocaleString()}</div>
                  </td>
                  <td>
                    <div className={`font-medium ${
                      campaign.openRate > 60 ? 'text-success-600' :
                      campaign.openRate > 40 ? 'text-warning-600' : 'text-danger-600'
                    }`}>
                      {campaign.openRate}%
                    </div>
                  </td>
                  <td>
                    <div className={`font-medium ${
                      campaign.clickRate > 30 ? 'text-success-600' :
                      campaign.clickRate > 15 ? 'text-warning-600' : 'text-danger-600'
                    }`}>
                      {campaign.clickRate}%
                    </div>
                  </td>
                  <td>
                    <div className={`font-medium ${
                      campaign.conversionRate > 8 ? 'text-success-600' :
                      campaign.conversionRate > 4 ? 'text-warning-600' : 'text-danger-600'
                    }`}>
                      {campaign.conversionRate}%
                    </div>
                  </td>
                  <td>
                    <div className="font-semibold text-success-600">
                      R$ {(campaign.revenue / 1000).toFixed(1)}k
                    </div>
                  </td>
                  <td>
                    <div className={`font-bold ${
                      campaign.roi > 200 ? 'text-success-600' :
                      campaign.roi > 100 ? 'text-warning-600' : 'text-danger-600'
                    }`}>
                      {campaign.roi}%
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Insights Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Insights de ML
            </h3>
          </div>
          <div className="card-body space-y-4">
            <div className="flex items-start space-x-3">
              <div className="h-2 w-2 bg-success-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm font-medium text-secondary-900">
                  Cluster "Novo Cliente" apresenta alta conversão
                </p>
                <p className="text-xs text-secondary-600 mt-1">
                  Taxa de conversão 3x maior que a média. Recomenda-se aumentar investimento.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="h-2 w-2 bg-warning-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm font-medium text-secondary-900">
                  Horário otimal: 16h-20h
                </p>
                <p className="text-xs text-secondary-600 mt-1">
                  Maior atividade de usuários detectada neste período.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="h-2 w-2 bg-primary-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm font-medium text-secondary-900">
                  Padrão sazonal identificado
                </p>
                <p className="text-xs text-secondary-600 mt-1">
                  Picos de atividade nas segundas e quartas-feiras.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Recomendações
            </h3>
          </div>
          <div className="card-body space-y-4">
            <div className="flex items-start space-x-3">
              <div className="p-1 bg-primary-100 rounded">
                <TrendingUp className="h-3 w-3 text-primary-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-secondary-900">
                  Aumentar frequência para Alto Valor
                </p>
                <p className="text-xs text-secondary-600 mt-1">
                  Cluster responde bem a campanhas mais frequentes.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="p-1 bg-warning-100 rounded">
                <Clock className="h-3 w-3 text-warning-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-secondary-900">
                  Ajustar timing de envios
                </p>
                <p className="text-xs text-secondary-600 mt-1">
                  Agendar campanhas para horários de maior engajamento.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="p-1 bg-success-100 rounded">
                <Target className="h-3 w-3 text-success-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-secondary-900">
                  Personalizar conteúdo por cluster
                </p>
                <p className="text-xs text-secondary-600 mt-1">
                  Cada cluster responde a diferentes tipos de conteúdo.
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default Analytics