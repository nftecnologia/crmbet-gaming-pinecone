import React from 'react'
import { motion } from 'framer-motion'
import { 
  Users, 
  Target, 
  Mail, 
  TrendingUp, 
  ArrowUpRight, 
  ArrowDownRight,
  Activity,
  Clock,
  CheckCircle
} from 'lucide-react'
import { Line, Doughnut, Bar } from 'react-chartjs-2'

const Dashboard = () => {
  // Mock data for metrics
  const metrics = [
    {
      title: 'Total de Usuários',
      value: '24,567',
      change: '+12.5%',
      trend: 'up',
      icon: Users,
      color: 'primary'
    },
    {
      title: 'Clusters Ativos',
      value: '8',
      change: '+2',
      trend: 'up',
      icon: Target,
      color: 'success'
    },
    {
      title: 'Campanhas Ativas',
      value: '12',
      change: '+3',
      trend: 'up',
      icon: Mail,
      color: 'warning'
    },
    {
      title: 'Taxa de Conversão',
      value: '4.2%',
      change: '-0.3%',
      trend: 'down',
      icon: TrendingUp,
      color: 'danger'
    }
  ]

  // Chart data
  const lineChartData = {
    labels: ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    datasets: [
      {
        label: 'Novos Usuários',
        data: [1200, 1900, 3000, 5000, 2000, 3000, 4500, 3200, 4800, 5200, 4100, 6000],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Conversões',
        data: [50, 80, 126, 210, 84, 126, 189, 134, 202, 218, 172, 252],
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4,
      }
    ]
  }

  const doughnutData = {
    labels: ['Alto Valor', 'Médio Valor', 'Baixo Valor', 'Novo Cliente', 'Inativo'],
    datasets: [
      {
        data: [35, 25, 20, 15, 5],
        backgroundColor: [
          '#3b82f6',
          '#22c55e',
          '#f59e0b',
          '#8b5cf6',
          '#ef4444'
        ],
        borderWidth: 0,
      }
    ]
  }

  const barChartData = {
    labels: ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
    datasets: [
      {
        label: 'Atividade dos Usuários',
        data: [4200, 3800, 4100, 4500, 3900, 2800, 2200],
        backgroundColor: '#3b82f6',
        borderRadius: 6,
      }
    ]
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: false,
      },
    },
  }

  const recentActivities = [
    {
      id: 1,
      type: 'campaign',
      title: 'Nova campanha "Black Friday 2024" criada',
      time: '2 minutos atrás',
      status: 'success'
    },
    {
      id: 2,
      type: 'cluster',
      title: 'Cluster "Alto Valor" atualizado com 15 novos usuários',
      time: '5 minutos atrás',
      status: 'info'
    },
    {
      id: 3,
      type: 'conversion',
      title: 'Meta de conversão da campanha "Reativação" atingida',
      time: '10 minutos atrás',
      status: 'success'
    },
    {
      id: 4,
      type: 'user',
      title: '1,234 novos usuários adicionados hoje',
      time: '1 hora atrás',
      status: 'info'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
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
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 card"
        >
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-secondary-900">
                Crescimento de Usuários & Conversões
              </h3>
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-3 h-3 bg-primary-500 rounded-full"></div>
                  <span className="text-secondary-600">Usuários</span>
                </div>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-3 h-3 bg-success-500 rounded-full"></div>
                  <span className="text-secondary-600">Conversões</span>
                </div>
              </div>
            </div>
          </div>
          <div className="card-body">
            <div className="h-80">
              <Line 
                data={lineChartData} 
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
                    },
                  },
                }} 
              />
            </div>
          </div>
        </motion.div>

        {/* Cluster Distribution */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Distribuição de Clusters
            </h3>
          </div>
          <div className="card-body">
            <div className="h-60">
              <Doughnut 
                data={doughnutData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                          size: 12,
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
        {/* Weekly Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-secondary-900">
              Atividade Semanal
            </h3>
          </div>
          <div className="card-body">
            <div className="h-60">
              <Bar 
                data={barChartData}
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
                    },
                  },
                }}
              />
            </div>
          </div>
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card"
        >
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-secondary-900">
                Atividade Recente
              </h3>
              <Activity className="h-5 w-5 text-secondary-400" />
            </div>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              {recentActivities.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className={`mt-0.5 h-2 w-2 rounded-full ${
                    activity.status === 'success' ? 'bg-success-500' : 'bg-primary-500'
                  }`}></div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-secondary-900">
                      {activity.title}
                    </p>
                    <div className="flex items-center mt-1 space-x-2">
                      <Clock className="h-3 w-3 text-secondary-400" />
                      <p className="text-xs text-secondary-500">{activity.time}</p>
                    </div>
                  </div>
                  <CheckCircle className={`h-4 w-4 ${
                    activity.status === 'success' ? 'text-success-500' : 'text-primary-500'
                  }`} />
                </div>
              ))}
            </div>
          </div>
          <div className="card-footer">
            <button className="w-full text-center text-sm text-primary-600 hover:text-primary-700 font-medium">
              Ver todas as atividades
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Dashboard