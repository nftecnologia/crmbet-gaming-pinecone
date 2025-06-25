# ⚙️ Tech Context - Ambiente de Desenvolvimento

## 🛠️ Stack Tecnológico Completo

### Backend Stack
```json
{
  "runtime": "Node.js 18+",
  "framework": "Express.js 4.x",
  "database": "PostgreSQL 15+",
  "cache": "Redis 7+",
  "queue": "RabbitMQ 3.12+",
  "orm": "Prisma ou Sequelize",
  "validation": "Joi",
  "testing": "Jest + Supertest",
  "documentation": "Swagger/OpenAPI"
}
```

### Frontend Stack
```json
{
  "framework": "React 18+",
  "styling": "Tailwind CSS 3+",
  "routing": "React Router 6+",
  "state": "Zustand ou Redux Toolkit",
  "forms": "React Hook Form",
  "charts": "Chart.js ou Recharts",
  "http": "Axios",
  "testing": "Jest + React Testing Library",
  "build": "Vite"
}
```

### ML/Data Stack
```python
{
  "language": "Python 3.9+",
  "ml_framework": "scikit-learn 1.3+",
  "data_processing": "pandas 2.0+",
  "numerical": "numpy 1.24+",
  "visualization": "matplotlib, seaborn",
  "job_scheduler": "Celery (opcional)",
  "testing": "pytest",
  "environment": "conda ou venv"
}
```

### DevOps/Infrastructure
```yaml
containerization:
  - Docker 24+
  - docker-compose 2.x
deployment:
  - Railway (primary)
  - Vercel (frontend fallback)
monitoring:
  - Railway logs
  - Custom health endpoints
ci_cd:
  - GitHub Actions (se necessário)
  - Railway auto-deploy
```

## 🔧 Configurações de Ambiente

### Variáveis de Ambiente Necessárias

#### Backend (.env)
```env
# Database
DATABASE_URL=postgresql://user:pass@host:port/db_name
REDIS_URL=redis://host:port

# API
PORT=3000
NODE_ENV=development|production
API_VERSION=v1

# External APIs
SMARTICO_API_URL=https://api.smartico.com
SMARTICO_API_KEY=your_api_key_here

# Security
JWT_SECRET=your_jwt_secret
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=100

# Queue
RABBITMQ_URL=amqp://user:pass@host:port

# ML Service
ML_SERVICE_URL=http://localhost:5000
```

#### Frontend (.env)
```env
# API Configuration
REACT_APP_API_URL=http://localhost:3000/api/v1
REACT_APP_WS_URL=ws://localhost:3000

# Environment
REACT_APP_ENV=development|production
REACT_APP_VERSION=1.0.0

# Features
REACT_APP_ENABLE_DEBUGGING=true
REACT_APP_MOCK_DATA=false
```

#### ML Pipeline (.env)
```env
# Database
DATABASE_URL=postgresql://user:pass@host:port/db_name

# Data Lake
DATA_LAKE_URL=s3://your-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# ML Configuration
ML_MODEL_PATH=./models
ML_BATCH_SIZE=1000
ML_CLUSTERING_ALGORITHM=kmeans

# Processing
CELERY_BROKER_URL=redis://host:port
CELERY_RESULT_BACKEND=redis://host:port
```

## 📦 Dependências Críticas

### Backend (package.json)
```json
{
  "dependencies": {
    "express": "^4.18.0",
    "pg": "^8.11.0",
    "redis": "^4.6.0",
    "amqplib": "^0.10.0",
    "joi": "^17.9.0",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "express-rate-limit": "^6.7.0",
    "swagger-jsdoc": "^6.2.0",
    "swagger-ui-express": "^4.6.0"
  },
  "devDependencies": {
    "jest": "^29.5.0",
    "supertest": "^6.3.0",
    "nodemon": "^2.0.0"
  }
}
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.11.0",
    "axios": "^1.4.0",
    "zustand": "^4.3.0",
    "react-hook-form": "^7.44.0",
    "chart.js": "^4.3.0",
    "react-chartjs-2": "^5.2.0",
    "tailwindcss": "^3.3.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^4.3.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/jest-dom": "^5.16.0"
  }
}
```

### ML Pipeline (requirements.txt)
```txt
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
psycopg2-binary==2.9.6
redis==4.5.5
python-dotenv==1.0.0
celery==5.2.7
pytest==7.3.1
```

## 🐳 Configuração Docker

### docker-compose.yml
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: crmbet
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3.12-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password

volumes:
  postgres_data:
```

## 🏗️ Scripts de Desenvolvimento

### Backend (package.json scripts)
```json
{
  "scripts": {
    "dev": "nodemon src/index.js",
    "start": "node src/index.js",
    "test": "jest",
    "test:watch": "jest --watch",
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "db:migrate": "npx prisma migrate dev",
    "db:seed": "node scripts/seed.js"
  }
}
```

### Frontend (package.json scripts)
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "jest",
    "lint": "eslint src/",
    "type-check": "tsc --noEmit"
  }
}
```

## 🚀 Comandos de Deploy

### Development
```bash
# Setup completo
docker-compose up -d
npm run dev          # Backend
npm run dev          # Frontend (nova janela)
python ml_cluster.py # ML Pipeline

# Testes
npm test            # Backend tests
npm test            # Frontend tests
pytest              # ML tests
```

### Production (Railway)
```bash
# Deploy automático via git push
git add .
git commit -m "Deploy to production"
git push origin main

# Ou deploy manual
railway up
```

## 🔍 Limitações Técnicas Conhecidas

### Performance
- **PostgreSQL**: Até 10k usuários simultâneos
- **Redis**: 1GB RAM máximo no free tier
- **Railway**: 512MB RAM, 1GB storage

### Integrações
- **Smartico API**: Rate limit de 1000 req/hour
- **Data Lake**: Dependente de provider externo
- **RabbitMQ**: Mensagens limitadas a 10MB

### Desenvolvimento
- **Hot Reload**: Pode ser lento com datasets grandes
- **ML Training**: Requer pelo menos 2GB RAM
- **Database**: Migrations podem ser demoradas

## 🛠️ Ferramentas de Desenvolvimento

### Recomendadas
- **IDE**: VS Code com extensões React/Node
- **Database Client**: PgAdmin ou DBeaver
- **API Testing**: Postman ou Insomnia
- **Git**: GitHub com branches feature/
- **Documentation**: Swagger UI integrado

### Configuração VS Code
```json
{
  "extensions": [
    "esbenp.prettier-vscode",
    "ms-python.python",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-json"
  ],
  "settings": {
    "editor.formatOnSave": true,
    "python.defaultInterpreterPath": "./venv/bin/python"
  }
}
```

---
*Manter atualizado conforme evolução do projeto*