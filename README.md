# 🚀 CRM Inteligente com Machine Learning

Sistema completo de CRM com clusterização ML para segmentação automática de usuários no setor de gaming/apostas, com integração Smartico e dashboard operacional.

## 🎯 Visão Geral

O **CRM Inteligente** utiliza algoritmos de Machine Learning para analisar comportamentos de usuários e criar clusters inteligentes para campanhas de marketing direcionadas. O sistema processa dados do Data Lake, executa clusterização automática e integra com CRMs externos como Smartico.

### 🏆 Características Principais

- **🤖 Machine Learning Avançado**: KMeans, DBSCAN, HDBSCAN com ensemble
- **📊 ETL Robusto**: Pipeline completo do Data Lake ao PostgreSQL
- **🎨 Dashboard Interativo**: React + Tailwind com visualizações avançadas
- **🔗 Integração CRM**: API REST completa com Smartico
- **🐳 Containerizado**: Docker com deployment Railway
- **🛡️ Segurança Enterprise**: JWT, rate limiting, validação rigorosa

## 🏗️ Arquitetura

```mermaid
graph TB
    subgraph "Data Sources"
        DL[Data Lake S3]
        TT[tbl_transactions]
    end
    
    subgraph "Processing Layer"
        ETL[ETL Pipeline]
        ML[ML Clustering]
    end
    
    subgraph "Storage Layer"
        PG[PostgreSQL]
        RD[Redis Cache]
        RMQ[RabbitMQ]
    end
    
    subgraph "API Layer"
        API[Backend API]
    end
    
    subgraph "Frontend Layer"
        UI[React Dashboard]
    end
    
    subgraph "External"
        CRM[Smartico CRM]
    end
    
    DL --> ETL
    TT --> ETL
    ETL --> PG
    PG --> ML
    ML --> PG
    API --> PG
    API --> RD
    API --> RMQ
    API --> CRM
    UI --> API
```

## 🚀 Quick Start

### Pré-requisitos

- Docker & Docker Compose
- Node.js 18+ (desenvolvimento)
- Python 3.9+ (desenvolvimento)

### 1. Clone e Configure

```bash
git clone <repository>
cd crmbet

# Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações
```

### 2. Inicie os Serviços

```bash
# Inicie todos os serviços
docker-compose up -d

# Aguarde os health checks
docker-compose ps
```

### 3. Acesse o Sistema

- **Dashboard**: http://localhost:3001
- **API Docs**: http://localhost:3000/api-docs
- **RabbitMQ**: http://localhost:15672
- **Database**: postgresql://localhost:5432/crmbet

## 📦 Serviços

### 🖥️ Backend API (Node.js)
- **Porta**: 3000
- **Tecnologias**: Express, PostgreSQL, Redis, RabbitMQ
- **Features**: JWT auth, rate limiting, Swagger docs

```bash
cd backend
npm install
npm run dev
```

### 🎨 Frontend Dashboard (React)
- **Porta**: 3001
- **Tecnologias**: React, Tailwind, Chart.js, Zustand
- **Features**: Dashboard, clusters, campanhas, analytics

```bash
cd frontend
npm install
npm run dev
```

### 🤖 ML Service (Python)
- **Tecnologias**: Scikit-learn, Pandas, NumPy
- **Algoritmos**: KMeans, DBSCAN, HDBSCAN, Ensemble

```bash
cd ml
pip install -r requirements.txt
python ml_cluster.py --mode train
```

### 🏗️ ETL Pipeline (Python)
- **Tecnologias**: Pandas, PostgreSQL, AWS S3
- **Features**: Data quality, feature engineering

```bash
cd etl
pip install -r requirements.txt
python run_pipeline.py --mode batch
```

## 📊 Features ML

### Algoritmos Implementados
- **KMeans**: Clustering centróide-based otimizado
- **DBSCAN**: Detecção de outliers e clusters densidade
- **HDBSCAN**: Clustering hierárquico com persistência
- **Ensemble**: Combinação de algoritmos com votação

### Features Geradas
- **Comportamentais**: Tipo jogo favorito, volatilidade apostas
- **Financeiras**: Ticket médio, CLV, padrões depósito/saque
- **Temporais**: Horários atividade, dias preferidos
- **RFM**: Recency, Frequency, Monetary analysis

### Clusters Típicos
- **High Roller Crash**: Gosta de crash games, apostas altas
- **Night Owl Casino**: Jogadores de cassino noturnos
- **Weekend Warrior**: Ativos principalmente fins de semana
- **VIP Live Dealer**: Preferência por jogos ao vivo
- **Sports Betting Pro**: Focados em apostas esportivas

## 🔗 API Endpoints

### Clusters
```bash
GET /api/v1/clusters               # Lista clusters
GET /api/v1/clusters/:id/users     # Usuários do cluster
GET /api/v1/clusters/:id/analytics # Analytics cluster
```

### Usuários
```bash
GET /api/v1/user/:id/segment       # Segmento ML usuário
GET /api/v1/users/search           # Busca usuários
GET /api/v1/users/:id/behavior     # Análise comportamental
```

### Campanhas
```bash
POST /api/v1/campaigns             # Criar campanha
GET /api/v1/campaigns/:id/results  # Resultados campanha
POST /api/v1/campaigns/:id/launch  # Lançar campanha
```

## 🐳 Docker

### Desenvolvimento
```bash
docker-compose up -d
```

### Produção
```bash
docker-compose -f docker-compose.production.yml up -d
```

### Serviços
- **PostgreSQL**: 5432
- **Redis**: 6379
- **RabbitMQ**: 5672, 15672 (management)
- **Backend**: 3000
- **Frontend**: 3001

## 🔧 Configuração

### Variáveis Principais (.env)
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis
REDIS_URL=redis://host:port

# Smartico
SMARTICO_API_URL=https://api.smartico.com
SMARTICO_API_KEY=your_key

# JWT
JWT_SECRET=your_secret

# Data Lake
AWS_ACCESS_KEY_ID=your_key
DATA_LAKE_BUCKET=your_bucket
```

## 📈 Monitoramento

### Health Checks
```bash
# Backend
curl http://localhost:3000/health

# Frontend
curl http://localhost:3001/health
```

### Logs
```bash
# Ver logs de todos os serviços
docker-compose logs -f

# Logs específicos
docker-compose logs -f backend
docker-compose logs -f ml_service
```

### Métricas
- **Response time**: P95 < 200ms
- **Accuracy**: Silhouette score > 0.6
- **Data quality**: Score A (>0.9)
- **Uptime**: 99.9%+

## 🧪 Testes

### Backend
```bash
cd backend
npm test
npm run test:coverage
```

### ML
```bash
cd ml
pytest tests/ -v
```

### E2E
```bash
cd frontend
npm run test:e2e
```

## 🚀 Deploy

### Railway (Recomendado)
```bash
# Conecte repositório GitHub
# Configure variáveis de ambiente
# Deploy automático via git push
```

### Manual
```bash
# Build images
docker-compose build

# Deploy
docker-compose -f docker-compose.production.yml up -d
```

## 🛡️ Segurança

### Implementado
- **Autenticação**: JWT com refresh tokens
- **Autorização**: Role-based access control
- **Rate Limiting**: 1000 req/h por IP
- **Validação**: Joi schemas + sanitização
- **Headers**: Helmet security headers
- **CORS**: Configuração restritiva

### Recomendações Produção
- Use HTTPS/TLS
- Configure firewall
- Monitore logs de segurança
- Atualize dependências regularmente
- Backup automático

## 📚 Documentação

### Estrutura
```
docs/
├── architecture/        # Arquitetura sistema
├── api/                # Especificação API
└── deployment/         # Guias deployment
```

### Swagger
Acesse http://localhost:3000/api-docs para documentação interativa da API.

## 🤝 Contribuição

1. Fork o projeto
2. Crie feature branch (`git checkout -b feature/amazing-feature`)
3. Commit mudanças (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Abra Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja `LICENSE` para detalhes.

## 🆘 Suporte

### Issues Comuns

**Q: Container não inicia**
```bash
# Verifique logs
docker-compose logs service_name

# Verifique variáveis de ambiente
cat .env
```

**Q: API retorna erro 500**
```bash
# Verifique conexão database
docker-compose logs postgres

# Verifique migrations
docker-compose exec backend npm run db:migrate
```

**Q: ML não gera clusters**
```bash
# Verifique dados
docker-compose exec ml_service python -c "import pandas as pd; print(pd.read_sql('SELECT count(*) FROM users', 'postgresql://...'))"

# Execute manual
docker-compose exec ml_service python ml_cluster.py --mode train
```

### Contato
- **Email**: suporte@crmbet.com
- **Docs**: https://docs.crmbet.com
- **Issues**: GitHub Issues

---

**🎯 CRM Inteligente - Transformando dados em insights, insights em resultados!**