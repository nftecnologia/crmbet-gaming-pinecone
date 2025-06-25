import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import {
  X,
  Save,
  Send,
  Calendar,
  Users,
  Target,
  Mail,
  Type,
  Image,
  Link,
  Eye,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react'
import toast from 'react-hot-toast'

// Validation schema
const campaignSchema = z.object({
  name: z.string().min(3, 'Nome deve ter pelo menos 3 caracteres'),
  description: z.string().min(10, 'Descrição deve ter pelo menos 10 caracteres'),
  type: z.enum(['promotional', 'reactivation', 'welcome', 'recommendation', 'survey', 'recovery']),
  cluster: z.string().min(1, 'Selecione um cluster'),
  subject: z.string().min(5, 'Assunto deve ter pelo menos 5 caracteres'),
  content: z.string().min(20, 'Conteúdo deve ter pelo menos 20 caracteres'),
  startDate: z.string().min(1, 'Data de início é obrigatória'),
  endDate: z.string().min(1, 'Data de fim é obrigatória'),
  frequency: z.enum(['once', 'daily', 'weekly', 'monthly']),
  sendTime: z.string().optional(),
})

const CampaignForm = ({ 
  isOpen = false, 
  onClose, 
  onSubmit, 
  initialData = null,
  mode = 'create' // 'create' or 'edit'
}) => {
  const [currentStep, setCurrentStep] = useState(1)
  const [previewMode, setPreviewMode] = useState(false)
  const [selectedCluster, setSelectedCluster] = useState(null)

  const { 
    register, 
    handleSubmit, 
    watch, 
    setValue, 
    formState: { errors, isSubmitting } 
  } = useForm({
    resolver: zodResolver(campaignSchema),
    defaultValues: initialData || {
      name: '',
      description: '',
      type: 'promotional',
      cluster: '',
      subject: '',
      content: '',
      startDate: '',
      endDate: '',
      frequency: 'once',
      sendTime: '09:00'
    }
  })

  const watchedValues = watch()

  // Mock data for clusters
  const clusters = [
    { id: 'alto-valor', name: 'Alto Valor', users: 3240, color: 'primary' },
    { id: 'medio-valor', name: 'Médio Valor', users: 5680, color: 'success' },
    { id: 'baixo-valor', name: 'Baixo Valor', users: 8920, color: 'warning' },
    { id: 'novo-cliente', name: 'Novo Cliente', users: 2150, color: 'info' },
    { id: 'vip-premium', name: 'VIP Premium', users: 450, color: 'purple' },
    { id: 'inativo', name: 'Inativo', users: 1890, color: 'danger' }
  ]

  const campaignTypes = [
    { value: 'promotional', label: 'Promocional', icon: Target, description: 'Campanhas de ofertas e promoções' },
    { value: 'reactivation', label: 'Reativação', icon: Users, description: 'Reativar usuários inativos' },
    { value: 'welcome', label: 'Boas-vindas', icon: Mail, description: 'Sequência para novos usuários' },
    { value: 'recommendation', label: 'Recomendação', icon: CheckCircle, description: 'Produtos personalizados' },
    { value: 'survey', label: 'Pesquisa', icon: Eye, description: 'Coleta de feedback' },
    { value: 'recovery', label: 'Recuperação', icon: AlertCircle, description: 'Carrinho abandonado' }
  ]

  const steps = [
    { id: 1, title: 'Configuração Básica', icon: Target },
    { id: 2, title: 'Segmentação', icon: Users },
    { id: 3, title: 'Conteúdo', icon: Mail },
    { id: 4, title: 'Agendamento', icon: Calendar },
    { id: 5, title: 'Revisão', icon: CheckCircle }
  ]

  const handleFormSubmit = async (data) => {
    try {
      await onSubmit?.(data)
      toast.success(`Campanha ${mode === 'create' ? 'criada' : 'atualizada'} com sucesso!`)
      onClose?.()
    } catch (error) {
      toast.error('Erro ao salvar campanha')
    }
  }

  const handleNext = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const getClusterColor = (color) => {
    const colorMap = {
      primary: 'bg-primary-100 text-primary-800 border-primary-200',
      success: 'bg-success-100 text-success-800 border-success-200',
      warning: 'bg-warning-100 text-warning-800 border-warning-200',
      danger: 'bg-danger-100 text-danger-800 border-danger-200',
      info: 'bg-sky-100 text-sky-800 border-sky-200',
      purple: 'bg-purple-100 text-purple-800 border-purple-200'
    }
    return colorMap[color] || colorMap.primary
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-secondary-900/50 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-secondary-200">
            <div>
              <h2 className="text-xl font-semibold text-secondary-900">
                {mode === 'create' ? 'Nova Campanha' : 'Editar Campanha'}
              </h2>
              <p className="text-sm text-secondary-600 mt-1">
                Crie campanhas segmentadas com base nos clusters de ML
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-secondary-100 rounded-lg transition-colors"
            >
              <X className="h-5 w-5 text-secondary-500" />
            </button>
          </div>

          {/* Progress Steps */}
          <div className="flex items-center justify-between p-6 bg-secondary-50 border-b border-secondary-200">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                  currentStep >= step.id
                    ? 'bg-primary-600 border-primary-600 text-white'
                    : 'bg-white border-secondary-300 text-secondary-400'
                }`}>
                  {currentStep > step.id ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <step.icon className="h-4 w-4" />
                  )}
                </div>
                <div className="ml-2 hidden sm:block">
                  <p className={`text-xs font-medium ${
                    currentStep >= step.id ? 'text-primary-600' : 'text-secondary-500'
                  }`}>
                    {step.title}
                  </p>
                </div>
                {index < steps.length - 1 && (
                  <div className={`hidden sm:block w-12 h-0.5 mx-4 ${
                    currentStep > step.id ? 'bg-primary-600' : 'bg-secondary-300'
                  }`} />
                )}
              </div>
            ))}
          </div>

          {/* Form Content */}
          <form onSubmit={handleSubmit(handleFormSubmit)} className="flex-1 overflow-y-auto">
            <div className="p-6 space-y-6">
              {/* Step 1: Basic Configuration */}
              {currentStep === 1 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <label className="form-label">Nome da Campanha</label>
                    <input
                      {...register('name')}
                      type="text"
                      className="form-input"
                      placeholder="Ex: Black Friday 2024"
                    />
                    {errors.name && (
                      <p className="form-error">{errors.name.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="form-label">Descrição</label>
                    <textarea
                      {...register('description')}
                      rows={3}
                      className="form-input"
                      placeholder="Descreva o objetivo e contexto da campanha..."
                    />
                    {errors.description && (
                      <p className="form-error">{errors.description.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="form-label">Tipo de Campanha</label>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {campaignTypes.map((type) => (
                        <label
                          key={type.value}
                          className={`p-3 border rounded-lg cursor-pointer transition-all ${
                            watchedValues.type === type.value
                              ? 'border-primary-500 bg-primary-50'
                              : 'border-secondary-300 hover:border-secondary-400'
                          }`}
                        >
                          <input
                            {...register('type')}
                            type="radio"
                            value={type.value}
                            className="sr-only"
                          />
                          <div className="flex items-center space-x-3">
                            <type.icon className={`h-5 w-5 ${
                              watchedValues.type === type.value
                                ? 'text-primary-600'
                                : 'text-secondary-400'
                            }`} />
                            <div>
                              <p className="font-medium text-secondary-900">{type.label}</p>
                              <p className="text-xs text-secondary-500">{type.description}</p>
                            </div>
                          </div>
                        </label>
                      ))}
                    </div>
                    {errors.type && (
                      <p className="form-error">{errors.type.message}</p>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 2: Segmentation */}
              {currentStep === 2 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <label className="form-label">Selecionar Cluster</label>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {clusters.map((cluster) => (
                        <label
                          key={cluster.id}
                          className={`p-4 border rounded-lg cursor-pointer transition-all ${
                            watchedValues.cluster === cluster.id
                              ? 'border-primary-500 bg-primary-50'
                              : 'border-secondary-300 hover:border-secondary-400'
                          }`}
                        >
                          <input
                            {...register('cluster')}
                            type="radio"
                            value={cluster.id}
                            className="sr-only"
                            onChange={(e) => {
                              setValue('cluster', e.target.value)
                              setSelectedCluster(cluster)
                            }}
                          />
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="flex items-center space-x-2">
                                <span className={`badge ${getClusterColor(cluster.color)}`}>
                                  {cluster.name}
                                </span>
                              </div>
                              <p className="text-sm text-secondary-600 mt-1">
                                {cluster.users.toLocaleString()} usuários
                              </p>
                            </div>
                            <div className={`w-4 h-4 border-2 rounded-full ${
                              watchedValues.cluster === cluster.id
                                ? 'border-primary-500 bg-primary-500'
                                : 'border-secondary-300'
                            }`}>
                              {watchedValues.cluster === cluster.id && (
                                <div className="w-full h-full rounded-full bg-white scale-50" />
                              )}
                            </div>
                          </div>
                        </label>
                      ))}
                    </div>
                    {errors.cluster && (
                      <p className="form-error">{errors.cluster.message}</p>
                    )}
                  </div>

                  {selectedCluster && (
                    <div className="p-4 bg-primary-50 rounded-lg border border-primary-200">
                      <h4 className="font-medium text-primary-900 mb-2">
                        Cluster Selecionado: {selectedCluster.name}
                      </h4>
                      <p className="text-sm text-primary-700">
                        Esta campanha será enviada para {selectedCluster.users.toLocaleString()} usuários
                        do cluster {selectedCluster.name}.
                      </p>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Step 3: Content */}
              {currentStep === 3 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-6"
                >
                  <div>
                    <label className="form-label">Assunto do Email</label>
                    <input
                      {...register('subject')}
                      type="text"
                      className="form-input"
                      placeholder="Ex: 🔥 Black Friday: Até 70% OFF em produtos selecionados!"
                    />
                    {errors.subject && (
                      <p className="form-error">{errors.subject.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="form-label">Conteúdo do Email</label>
                    <textarea
                      {...register('content')}
                      rows={12}
                      className="form-input font-mono text-sm"
                      placeholder="Olá {{nome}},&#10;&#10;Temos uma oferta especial para você!&#10;&#10;[Conteúdo da campanha]&#10;&#10;Atenciosamente,&#10;Equipe CRM ML"
                    />
                    {errors.content && (
                      <p className="form-error">{errors.content.message}</p>
                    )}
                    <p className="text-xs text-secondary-500 mt-2">
                      Use {{nome}} para personalizar com o nome do usuário
                    </p>
                  </div>

                  <div className="flex items-center space-x-3">
                    <button
                      type="button"
                      onClick={() => setPreviewMode(!previewMode)}
                      className="btn-ghost btn-sm"
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      {previewMode ? 'Editar' : 'Visualizar'}
                    </button>
                  </div>

                  {previewMode && (
                    <div className="p-4 bg-secondary-50 rounded-lg border border-secondary-200">
                      <h4 className="font-medium text-secondary-900 mb-3">Preview do Email</h4>
                      <div className="bg-white p-4 rounded border">
                        <div className="mb-3 pb-3 border-b border-secondary-200">
                          <p className="font-medium text-secondary-900">
                            Assunto: {watchedValues.subject || 'Sem assunto'}
                          </p>
                        </div>
                        <div className="whitespace-pre-wrap text-sm text-secondary-700">
                          {watchedValues.content?.replace('{{nome}}', 'João Silva') || 'Conteúdo vazio'}
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Step 4: Scheduling */}
              {currentStep === 4 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-6"
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    <div>
                      <label className="form-label">Data de Início</label>
                      <input
                        {...register('startDate')}
                        type="datetime-local"
                        className="form-input"
                      />
                      {errors.startDate && (
                        <p className="form-error">{errors.startDate.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="form-label">Data de Fim</label>
                      <input
                        {...register('endDate')}
                        type="datetime-local"
                        className="form-input"
                      />
                      {errors.endDate && (
                        <p className="form-error">{errors.endDate.message}</p>
                      )}
                    </div>
                  </div>

                  <div>
                    <label className="form-label">Frequência de Envio</label>
                    <select {...register('frequency')} className="form-input">
                      <option value="once">Envio único</option>
                      <option value="daily">Diário</option>
                      <option value="weekly">Semanal</option>
                      <option value="monthly">Mensal</option>
                    </select>
                  </div>

                  <div>
                    <label className="form-label">Horário Preferencial (opcional)</label>
                    <input
                      {...register('sendTime')}
                      type="time"
                      className="form-input max-w-xs"
                    />
                    <p className="text-xs text-secondary-500 mt-1">
                      Baseado no horário de maior atividade do cluster selecionado
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Step 5: Review */}
              {currentStep === 5 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-6"
                >
                  <div className="bg-secondary-50 p-6 rounded-lg">
                    <h3 className="text-lg font-semibold text-secondary-900 mb-4">
                      Resumo da Campanha
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-secondary-500">Nome:</p>
                        <p className="font-medium text-secondary-900">{watchedValues.name}</p>
                      </div>
                      <div>
                        <p className="text-secondary-500">Tipo:</p>
                        <p className="font-medium text-secondary-900">
                          {campaignTypes.find(t => t.value === watchedValues.type)?.label}
                        </p>
                      </div>
                      <div>
                        <p className="text-secondary-500">Cluster:</p>
                        <p className="font-medium text-secondary-900">
                          {clusters.find(c => c.id === watchedValues.cluster)?.name}
                        </p>
                      </div>
                      <div>
                        <p className="text-secondary-500">Usuários Alvo:</p>
                        <p className="font-medium text-secondary-900">
                          {clusters.find(c => c.id === watchedValues.cluster)?.users.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-secondary-500">Data de Início:</p>
                        <p className="font-medium text-secondary-900">
                          {watchedValues.startDate ? new Date(watchedValues.startDate).toLocaleString() : 'Não definida'}
                        </p>
                      </div>
                      <div>
                        <p className="text-secondary-500">Frequência:</p>
                        <p className="font-medium text-secondary-900">
                          {watchedValues.frequency === 'once' ? 'Envio único' : 
                           watchedValues.frequency === 'daily' ? 'Diário' :
                           watchedValues.frequency === 'weekly' ? 'Semanal' : 'Mensal'}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-warning-50 p-4 rounded-lg border border-warning-200">
                    <div className="flex items-start space-x-3">
                      <AlertCircle className="h-5 w-5 text-warning-600 mt-0.5" />
                      <div>
                        <h4 className="font-medium text-warning-900">Importante</h4>
                        <p className="text-sm text-warning-700 mt-1">
                          Verifique todos os dados antes de criar a campanha. 
                          Após criada, alguns campos não poderão ser editados.
                        </p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between p-6 border-t border-secondary-200 bg-secondary-50">
              <div className="flex items-center space-x-3">
                {currentStep > 1 && (
                  <button
                    type="button"
                    onClick={handlePrevious}
                    className="btn-secondary btn-md"
                  >
                    Anterior
                  </button>
                )}
              </div>

              <div className="flex items-center space-x-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="btn-ghost btn-md"
                >
                  Cancelar
                </button>
                {currentStep < steps.length ? (
                  <button
                    type="button"
                    onClick={handleNext}
                    className="btn-primary btn-md"
                  >
                    Próximo
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="btn-primary btn-md"
                  >
                    {isSubmitting ? (
                      <>
                        <Clock className="h-4 w-4 mr-2 animate-spin" />
                        Salvando...
                      </>
                    ) : (
                      <>
                        <Save className="h-4 w-4 mr-2" />
                        {mode === 'create' ? 'Criar Campanha' : 'Salvar Alterações'}
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          </form>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

export default CampaignForm