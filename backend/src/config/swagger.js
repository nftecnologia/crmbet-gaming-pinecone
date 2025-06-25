/**
 * Swagger Configuration
 * 
 * Configuração completa do Swagger/OpenAPI 3.0
 * para documentação automática da API
 * 
 * @author CRM Team
 */

const swaggerConfig = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'CRM Inteligente - API REST',
      version: '1.0.0',
      description: `
        API REST robusta para CRM com Machine Learning e integração Smartico.
        
        ## Recursos Principais
        - Segmentação de usuários com ML
        - Clustering automático de clientes
        - Campanhas personalizadas
        - Integração Smartico CRM
        - Analytics em tempo real
        
        ## Autenticação
        Todas as rotas da API (exceto webhooks) requerem autenticação JWT.
        
        ## Rate Limiting
        - Global: 1000 requests/hora por IP
        - API: 500 requests/hora por usuário
        - Webhooks: 100 requests/minuto por IP
        - Campanhas: 50 requests/hora por usuário
      `,
      contact: {
        name: 'CRM Team',
        email: 'dev@crmbet.com'
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT'
      }
    },
    servers: [
      {
        url: process.env.API_BASE_URL || 'http://localhost:3000',
        description: 'Development server'
      },
      {
        url: 'https://api.crmbet.com',
        description: 'Production server'
      }
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
          description: 'JWT token obtido através do login'
        }
      },
      schemas: {
        // User schemas
        User: {
          type: 'object',
          required: ['id', 'external_id', 'email', 'name'],
          properties: {
            id: {
              type: 'integer',
              description: 'ID interno do usuário'
            },
            external_id: {
              type: 'string',
              description: 'ID externo (Smartico)'
            },
            email: {
              type: 'string',
              format: 'email',
              description: 'Email do usuário'
            },
            name: {
              type: 'string',
              description: 'Nome do usuário'
            },
            segment: {
              type: 'string',
              enum: ['high_value', 'medium_value', 'low_value', 'new_user', 'inactive'],
              description: 'Segmento ML do usuário'
            },
            cluster_id: {
              type: 'integer',
              nullable: true,
              description: 'ID do cluster ML'
            },
            registration_date: {
              type: 'string',
              format: 'date-time',
              description: 'Data de registro'
            },
            last_activity: {
              type: 'string',
              format: 'date-time',
              nullable: true,
              description: 'Última atividade'
            },
            total_deposits: {
              type: 'number',
              format: 'decimal',
              description: 'Total de depósitos'
            },
            total_withdrawals: {
              type: 'number',
              format: 'decimal',
              description: 'Total de saques'
            },
            total_bets: {
              type: 'number',
              format: 'decimal',
              description: 'Total apostado'
            },
            bet_count: {
              type: 'integer',
              description: 'Número de apostas'
            },
            win_rate: {
              type: 'number',
              format: 'decimal',
              description: 'Taxa de vitórias (%)'
            },
            metadata: {
              type: 'object',
              description: 'Metadados adicionais'
            }
          }
        },
        
        // Cluster schemas
        Cluster: {
          type: 'object',
          required: ['id', 'name', 'algorithm'],
          properties: {
            id: {
              type: 'integer',
              description: 'ID do cluster'
            },
            name: {
              type: 'string',
              description: 'Nome do cluster'
            },
            description: {
              type: 'string',
              nullable: true,
              description: 'Descrição do cluster'
            },
            algorithm: {
              type: 'string',
              enum: ['kmeans', 'dbscan', 'hierarchical'],
              description: 'Algoritmo utilizado'
            },
            parameters: {
              type: 'object',
              description: 'Parâmetros do algoritmo'
            },
            features: {
              type: 'object',
              description: 'Features utilizadas'
            },
            user_count: {
              type: 'integer',
              description: 'Número de usuários no cluster'
            },
            created_at: {
              type: 'string',
              format: 'date-time',
              description: 'Data de criação'
            }
          }
        },
        
        // Campaign schemas
        Campaign: {
          type: 'object',
          required: ['name', 'type', 'content'],
          properties: {
            id: {
              type: 'integer',
              description: 'ID da campanha'
            },
            name: {
              type: 'string',
              minLength: 3,
              maxLength: 255,
              description: 'Nome da campanha'
            },
            description: {
              type: 'string',
              nullable: true,
              description: 'Descrição da campanha'
            },
            type: {
              type: 'string',
              enum: ['email', 'sms', 'push', 'in_app'],
              description: 'Tipo da campanha'
            },
            status: {
              type: 'string',
              enum: ['draft', 'scheduled', 'running', 'completed', 'cancelled'],
              description: 'Status da campanha'
            },
            target_cluster_id: {
              type: 'integer',
              nullable: true,
              description: 'ID do cluster alvo'
            },
            target_segment: {
              type: 'string',
              nullable: true,
              description: 'Segmento alvo'
            },
            target_criteria: {
              type: 'object',
              description: 'Critérios de segmentação'
            },
            content: {
              type: 'object',
              required: ['subject', 'message'],
              properties: {
                subject: {
                  type: 'string',
                  description: 'Assunto/título'
                },
                message: {
                  type: 'string',
                  description: 'Conteúdo da mensagem'
                },
                template_id: {
                  type: 'string',
                  nullable: true,
                  description: 'ID do template'
                }
              }
            },
            schedule_at: {
              type: 'string',
              format: 'date-time',
              nullable: true,
              description: 'Agendamento'
            },
            smartico_campaign_id: {
              type: 'string',
              nullable: true,
              description: 'ID da campanha no Smartico'
            },
            total_sent: {
              type: 'integer',
              description: 'Total enviado'
            },
            total_opened: {
              type: 'integer',
              description: 'Total aberto'
            },
            total_clicked: {
              type: 'integer',
              description: 'Total clicado'
            },
            total_converted: {
              type: 'integer',
              description: 'Total convertido'
            }
          }
        },
        
        // Campaign Create Request
        CampaignCreateRequest: {
          type: 'object',
          required: ['name', 'type', 'content'],
          properties: {
            name: {
              type: 'string',
              minLength: 3,
              maxLength: 255,
              example: 'Promoção de Boas-vindas'
            },
            description: {
              type: 'string',
              example: 'Campanha para novos usuários'
            },
            type: {
              type: 'string',
              enum: ['email', 'sms', 'push', 'in_app'],
              example: 'email'
            },
            target_cluster_id: {
              type: 'integer',
              nullable: true,
              example: 1
            },
            target_segment: {
              type: 'string',
              nullable: true,
              example: 'new_user'
            },
            target_criteria: {
              type: 'object',
              example: {
                min_deposits: 0,
                max_days_since_registration: 7
              }
            },
            content: {
              type: 'object',
              required: ['subject', 'message'],
              properties: {
                subject: {
                  type: 'string',
                  example: 'Bem-vindo ao CRM!'
                },
                message: {
                  type: 'string',
                  example: 'Aproveite nossa oferta especial...'
                },
                template_id: {
                  type: 'string',
                  nullable: true,
                  example: 'welcome_template'
                }
              }
            },
            schedule_at: {
              type: 'string',
              format: 'date-time',
              nullable: true,
              example: '2024-01-01T12:00:00Z'
            }
          }
        },
        
        // Campaign Results
        CampaignResults: {
          type: 'object',
          properties: {
            campaign_id: {
              type: 'integer'
            },
            total_sent: {
              type: 'integer'
            },
            total_opened: {
              type: 'integer'
            },
            total_clicked: {
              type: 'integer'
            },
            total_converted: {
              type: 'integer'
            },
            open_rate: {
              type: 'number',
              format: 'decimal'
            },
            click_rate: {
              type: 'number',
              format: 'decimal'
            },
            conversion_rate: {
              type: 'number',
              format: 'decimal'
            },
            results: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  user_id: {
                    type: 'integer'
                  },
                  status: {
                    type: 'string',
                    enum: ['sent', 'opened', 'clicked', 'converted', 'failed']
                  },
                  sent_at: {
                    type: 'string',
                    format: 'date-time'
                  },
                  opened_at: {
                    type: 'string',
                    format: 'date-time',
                    nullable: true
                  },
                  clicked_at: {
                    type: 'string',
                    format: 'date-time',
                    nullable: true
                  },
                  converted_at: {
                    type: 'string',
                    format: 'date-time',
                    nullable: true
                  },
                  conversion_value: {
                    type: 'number',
                    format: 'decimal',
                    nullable: true
                  }
                }
              }
            }
          }
        },
        
        // Health Check Response
        HealthCheck: {
          type: 'object',
          properties: {
            status: {
              type: 'string',
              enum: ['healthy', 'unhealthy']
            },
            timestamp: {
              type: 'string',
              format: 'date-time'
            },
            version: {
              type: 'string'
            },
            environment: {
              type: 'string'
            },
            services: {
              type: 'object',
              properties: {
                database: {
                  type: 'object',
                  properties: {
                    status: { type: 'string' },
                    latency: { type: 'string' }
                  }
                },
                redis: {
                  type: 'object',
                  properties: {
                    status: { type: 'string' },
                    latency: { type: 'string' }
                  }
                },
                rabbitmq: {
                  type: 'object',
                  properties: {
                    status: { type: 'string' }
                  }
                }
              }
            }
          }
        },
        
        // Error Response
        Error: {
          type: 'object',
          properties: {
            error: {
              type: 'string',
              description: 'Tipo do erro'
            },
            message: {
              type: 'string',
              description: 'Mensagem de erro'
            },
            details: {
              type: 'object',
              description: 'Detalhes adicionais'
            },
            timestamp: {
              type: 'string',
              format: 'date-time',
              description: 'Timestamp do erro'
            }
          }
        }
      },
      responses: {
        Unauthorized: {
          description: 'Token de acesso inválido ou ausente',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error'
              }
            }
          }
        },
        Forbidden: {
          description: 'Acesso negado',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error'
              }
            }
          }
        },
        NotFound: {
          description: 'Recurso não encontrado',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error'
              }
            }
          }
        },
        ValidationError: {
          description: 'Erro de validação',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error'
              }
            }
          }
        },
        RateLimit: {
          description: 'Rate limit excedido',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error'
              }
            }
          }
        },
        ServerError: {
          description: 'Erro interno do servidor',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error'
              }
            }
          }
        }
      }
    },
    security: [
      {
        bearerAuth: []
      }
    ],
    tags: [
      {
        name: 'Health',
        description: 'Health checks e status do sistema'
      },
      {
        name: 'Users',
        description: 'Gestão de usuários e segmentação'
      },
      {
        name: 'Clusters',
        description: 'Clusters de Machine Learning'
      },
      {
        name: 'Campaigns',
        description: 'Campanhas de marketing'
      },
      {
        name: 'Webhooks',
        description: 'Webhooks de integração'
      }
    ]
  },
  apis: [
    './src/controllers/*.js',
    './src/routes/*.js',
    './src/index.js'
  ]
};

module.exports = swaggerConfig;