# CRM Inteligente - Infrastructure Deployment Guide

## Overview

This document provides comprehensive infrastructure deployment guidelines for CRM Inteligente, covering production deployment on Railway, environment configurations, CI/CD pipelines, security hardening, and operational procedures.

## Infrastructure Architecture

### Production Environment Topology

```mermaid
graph TB
    %% External Services
    Internet[Internet Users]
    CDN[Cloudflare CDN<br/>Global Edge Cache]
    DNS[Cloudflare DNS<br/>Load Balancing]
    
    %% Railway Infrastructure
    subgraph "Railway Production Environment"
        LB[Railway Load Balancer<br/>Auto-scaling]
        
        subgraph "Application Services"
            API1[Backend API Instance 1<br/>Node.js]
            API2[Backend API Instance 2<br/>Node.js]
            ML1[ML Service Instance 1<br/>Python]
            ML2[ML Service Instance 2<br/>Python]
            ETL[ETL Service<br/>Python Cron Jobs]
        end
        
        subgraph "Data Layer"
            PrimaryDB[(PostgreSQL Primary<br/>Railway Postgres)]
            ReadReplica[(PostgreSQL Read Replica<br/>Analytics)]
            Redis[(Redis Cluster<br/>Railway Redis)]
            RabbitMQ[RabbitMQ<br/>Message Queue]
        end
        
        subgraph "Monitoring & Logging"
            Prometheus[Prometheus<br/>Metrics]
            Grafana[Grafana<br/>Dashboards]
            ELK[ELK Stack<br/>Centralized Logging]
        end
    end
    
    %% External Dependencies
    SmartICO[SmartICO API]
    PaymentGW[Payment Gateways]
    EmailService[SendGrid/Mailgun]
    SMSService[Twilio SMS]
    
    %% Data Flow
    Internet --> CDN
    CDN --> DNS
    DNS --> LB
    LB --> API1
    LB --> API2
    API1 --> PrimaryDB
    API2 --> PrimaryDB
    API1 --> ReadReplica
    API2 --> ReadReplica
    API1 --> Redis
    API2 --> Redis
    API1 --> RabbitMQ
    API2 --> RabbitMQ
    
    ML1 --> PrimaryDB
    ML2 --> ReadReplica
    ML1 --> Redis
    ML2 --> Redis
    ML1 --> RabbitMQ
    ML2 --> RabbitMQ
    
    ETL --> PrimaryDB
    ETL --> SmartICO
    
    API1 --> SmartICO
    API2 --> PaymentGW
    API1 --> EmailService
    API2 --> SMSService
    
    API1 --> Prometheus
    API2 --> Prometheus
    ML1 --> Prometheus
    ML2 --> Prometheus
    Prometheus --> Grafana
    
    API1 --> ELK
    API2 --> ELK
    ML1 --> ELK
    ML2 --> ELK
```

## Railway Deployment Configuration

### Railway Project Structure

```yaml
# railway.toml
[build]
builder = "DOCKER"
buildCommand = "docker build -t crmbet-backend ."

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[environments.production]
[environments.production.variables]
NODE_ENV = "production"
LOG_LEVEL = "info"
RAILWAY_ENVIRONMENT = "production"

[environments.staging]
[environments.staging.variables]
NODE_ENV = "staging"
LOG_LEVEL = "debug"
RAILWAY_ENVIRONMENT = "staging"
```

### Service Deployment Configurations

#### Backend API Service
```dockerfile
# backend/Dockerfile.production
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY src/ ./src/

# Build application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S crmbet -u 1001

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Set ownership
RUN chown -R crmbet:nodejs /app
USER crmbet

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000

CMD ["node", "dist/index.js"]
```

#### ML Service Configuration
```dockerfile
# ml/Dockerfile.production
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user
RUN groupadd -r mlservice && useradd -r -g mlservice mlservice

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Set ownership
RUN chown -R mlservice:mlservice /app
USER mlservice

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Environment Configuration

#### Production Environment Variables
```bash
# .env.production
# Application
NODE_ENV=production
PORT=3000
API_VERSION=v1
LOG_LEVEL=info

# Database (Railway PostgreSQL)
DATABASE_URL=postgresql://user:password@host:port/database
DB_SSL=true
DB_POOL_MIN=10
DB_POOL_MAX=100
DB_POOL_IDLE_TIMEOUT=600000
DB_POOL_ACQUIRE_TIMEOUT=30000

# Redis (Railway Redis)
REDIS_URL=redis://default:password@host:port
REDIS_TLS=true
REDIS_MAX_RETRIES=3
REDIS_RETRY_DELAY=100

# Message Queue
RABBITMQ_URL=amqps://user:password@host:port/vhost
RABBITMQ_SSL=true
RABBITMQ_PREFETCH=10

# Security
JWT_SECRET=${RAILWAY_VARIABLE_JWT_SECRET}
JWT_EXPIRES_IN=15m
JWT_REFRESH_EXPIRES_IN=7d
BCRYPT_ROUNDS=12
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=1000

# External APIs
SMARTICO_API_URL=https://api.smartico.com
SMARTICO_API_KEY=${RAILWAY_VARIABLE_SMARTICO_API_KEY}
PAYMENT_API_URL=https://api.payment.com
PAYMENT_API_KEY=${RAILWAY_VARIABLE_PAYMENT_API_KEY}

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_TIMEOUT=30000

# ML Service
ML_SERVICE_URL=https://ml-service.railway.app
ML_BATCH_SIZE=1000
ML_CACHE_TTL=3600

# Email Service
SENDGRID_API_KEY=${RAILWAY_VARIABLE_SENDGRID_API_KEY}
EMAIL_FROM=noreply@crmbet.com

# SMS Service
TWILIO_ACCOUNT_SID=${RAILWAY_VARIABLE_TWILIO_ACCOUNT_SID}
TWILIO_AUTH_TOKEN=${RAILWAY_VARIABLE_TWILIO_AUTH_TOKEN}

# File Storage
AWS_ACCESS_KEY_ID=${RAILWAY_VARIABLE_AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${RAILWAY_VARIABLE_AWS_SECRET_ACCESS_KEY}
AWS_REGION=us-east-1
S3_BUCKET=crmbet-production

# CDN
CLOUDFLARE_ZONE_ID=${RAILWAY_VARIABLE_CLOUDFLARE_ZONE_ID}
CLOUDFLARE_API_TOKEN=${RAILWAY_VARIABLE_CLOUDFLARE_API_TOKEN}
CDN_URL=https://cdn.crmbet.com

# Security Headers
CORS_ORIGIN=https://app.crmbet.com,https://admin.crmbet.com
CORS_CREDENTIALS=true
HELMET_CSP_ENABLED=true
```

#### Staging Environment Variables
```bash
# .env.staging
# Application
NODE_ENV=staging
PORT=3000
API_VERSION=v1
LOG_LEVEL=debug

# Database (Railway PostgreSQL - Staging)
DATABASE_URL=postgresql://user:password@staging-host:port/database
DB_SSL=true
DB_POOL_MIN=5
DB_POOL_MAX=20

# Redis (Railway Redis - Staging)
REDIS_URL=redis://default:password@staging-host:port
REDIS_TLS=true

# Reduced security for testing
JWT_SECRET=staging-jwt-secret-key
RATE_LIMIT_MAX=10000

# External APIs (Sandbox)
SMARTICO_API_URL=https://sandbox-api.smartico.com
SMARTICO_API_KEY=${RAILWAY_VARIABLE_SMARTICO_STAGING_KEY}

# File Storage (Staging bucket)
S3_BUCKET=crmbet-staging

# CORS (Allow localhost for development)
CORS_ORIGIN=http://localhost:3001,https://staging.crmbet.com
```

## CI/CD Pipeline Configuration

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to Railway

on:
  push:
    branches:
      - main
      - staging
  pull_request:
    branches:
      - main

env:
  RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: crmbet_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: backend/package-lock.json
      
      - name: Install backend dependencies
        working-directory: ./backend
        run: npm ci
      
      - name: Run linting
        working-directory: ./backend
        run: npm run lint
      
      - name: Run type checking
        working-directory: ./backend
        run: npm run type-check
      
      - name: Run unit tests
        working-directory: ./backend
        run: npm run test
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/crmbet_test
          REDIS_URL: redis://localhost:6379
      
      - name: Run integration tests
        working-directory: ./backend
        run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/crmbet_test
          REDIS_URL: redis://localhost:6379
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install ML service dependencies
        working-directory: ./ml
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run ML service tests
        working-directory: ./ml
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          files: ./backend/coverage/lcov.info,./ml/coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Snyk security scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  deploy-staging:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    if: github.ref == 'refs/heads/staging'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Railway CLI
        run: npm install -g @railway/cli
      
      - name: Deploy to staging
        run: |
          railway login --token $RAILWAY_TOKEN
          railway environment staging
          railway up --service backend
          railway up --service ml-service
          railway up --service etl-service
      
      - name: Run smoke tests
        run: |
          sleep 60  # Wait for deployment
          curl -f https://staging-api.crmbet.com/health
          curl -f https://staging-ml.crmbet.com/health
      
      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}

  deploy-production:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Railway CLI
        run: npm install -g @railway/cli
      
      - name: Deploy to production
        run: |
          railway login --token $RAILWAY_TOKEN
          railway environment production
          
          # Deploy services one by one with health checks
          railway up --service backend
          sleep 30
          curl -f https://api.crmbet.com/health
          
          railway up --service ml-service
          sleep 30
          curl -f https://ml.crmbet.com/health
          
          railway up --service etl-service
      
      - name: Run production smoke tests
        run: |
          ./scripts/smoke-tests.sh
      
      - name: Notify production deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          channel: '#critical'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          custom_payload: |
            {
              "text": "🚀 Production deployment completed successfully",
              "attachments": [{
                "color": "good",
                "fields": [{
                  "title": "Environment",
                  "value": "Production",
                  "short": true
                }, {
                  "title": "Version",
                  "value": "${{ github.sha }}",
                  "short": true
                }]
              }]
            }

  rollback:
    runs-on: ubuntu-latest
    if: failure() && github.ref == 'refs/heads/main'
    
    steps:
      - name: Rollback production deployment
        run: |
          railway login --token $RAILWAY_TOKEN
          railway environment production
          railway rollback
      
      - name: Notify rollback
        uses: 8398a7/action-slack@v3
        with:
          status: 'failure'
          channel: '#critical'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          custom_payload: |
            {
              "text": "❌ Production deployment failed - Rolling back",
              "attachments": [{
                "color": "danger",
                "fields": [{
                  "title": "Action",
                  "value": "Automatic rollback initiated",
                  "short": true
                }]
              }]
            }
```

### Deployment Scripts

#### Smoke Tests Script
```bash
#!/bin/bash
# scripts/smoke-tests.sh

set -e

API_URL="${API_URL:-https://api.crmbet.com}"
ML_URL="${ML_URL:-https://ml.crmbet.com}"

echo "Running smoke tests..."

# Test API health
echo "Testing API health..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [ $response -eq 200 ]; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed (HTTP $response)"
    exit 1
fi

# Test API authentication
echo "Testing API authentication..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/v1/auth/status")
if [ $response -eq 401 ]; then
    echo "✅ API authentication working"
else
    echo "❌ API authentication failed (HTTP $response)"
    exit 1
fi

# Test ML service health
echo "Testing ML service health..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$ML_URL/health")
if [ $response -eq 200 ]; then
    echo "✅ ML service health check passed"
else
    echo "❌ ML service health check failed (HTTP $response)"
    exit 1
fi

# Test database connectivity
echo "Testing database connectivity..."
response=$(curl -s "$API_URL/health/detailed" | jq -r '.checks.database.status')
if [ "$response" = "healthy" ]; then
    echo "✅ Database connectivity check passed"
else
    echo "❌ Database connectivity check failed"
    exit 1
fi

# Test Redis connectivity
echo "Testing Redis connectivity..."
response=$(curl -s "$API_URL/health/detailed" | jq -r '.checks.redis.status')
if [ "$response" = "healthy" ]; then
    echo "✅ Redis connectivity check passed"
else
    echo "❌ Redis connectivity check failed"
    exit 1
fi

echo "🎉 All smoke tests passed!"
```

#### Database Migration Script
```bash
#!/bin/bash
# scripts/migrate.sh

set -e

echo "Running database migrations..."

# Check if database is accessible
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; then
    echo "❌ Database is not accessible"
    exit 1
fi

# Run migrations
echo "Applying schema migrations..."
psql "$DATABASE_URL" -f docs/sql/01_schema.sql

echo "Creating indexes..."
psql "$DATABASE_URL" -f docs/sql/02_indexes.sql

echo "Loading initial data..."
psql "$DATABASE_URL" -f docs/sql/03_initial_data.sql

echo "✅ Database migrations completed successfully"
```

## Security Hardening

### Railway Security Configuration

#### Environment Variable Management
```bash
# Set sensitive variables using Railway CLI
railway variables set JWT_SECRET=$(openssl rand -base64 32)
railway variables set DB_PASSWORD=$(openssl rand -base64 32)
railway variables set REDIS_PASSWORD=$(openssl rand -base64 32)
railway variables set ENCRYPTION_KEY=$(openssl rand -base64 32)

# Set API keys
railway variables set SMARTICO_API_KEY="your-smartico-api-key"
railway variables set SENDGRID_API_KEY="your-sendgrid-api-key"
railway variables set TWILIO_AUTH_TOKEN="your-twilio-token"

# Set AWS credentials
railway variables set AWS_ACCESS_KEY_ID="your-aws-access-key"
railway variables set AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
```

#### Network Security
```yaml
# railway-security.yml
security:
  # Enable HTTPS only
  forceHTTPS: true
  
  # Security headers
  headers:
    - name: "Strict-Transport-Security"
      value: "max-age=31536000; includeSubDomains; preload"
    - name: "X-Content-Type-Options"
      value: "nosniff"
    - name: "X-Frame-Options"
      value: "DENY"
    - name: "X-XSS-Protection"
      value: "1; mode=block"
    - name: "Referrer-Policy"
      value: "strict-origin-when-cross-origin"
  
  # Rate limiting
  rateLimiting:
    enabled: true
    requests: 1000
    window: "15m"
  
  # IP allowlist for admin endpoints
  ipAllowlist:
    - "203.0.113.0/24"  # Office IP range
    - "198.51.100.0/24" # VPN IP range
```

### Container Security Hardening
```dockerfile
# Security-hardened Dockerfile
FROM node:18-alpine

# Security updates
RUN apk update && apk upgrade && apk add --no-cache dumb-init

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S crmbet -u 1001

# Set working directory
WORKDIR /app

# Copy and install dependencies as root
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY --chown=crmbet:nodejs . .

# Remove unnecessary packages
RUN apk del apk-tools

# Switch to non-root user
USER crmbet

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Security labels
LABEL security.scan="2024-01" \
      security.level="high" \
      maintainer="security@crmbet.com"

CMD ["node", "dist/index.js"]
```

## Monitoring & Observability

### Production Monitoring Stack

#### Prometheus Configuration for Railway
```yaml
# prometheus-railway.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'railway-production'
    environment: 'production'

scrape_configs:
  - job_name: 'crmbet-backend'
    railway_sd_configs:
      - service: 'backend'
        environment: 'production'
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'crmbet-ml'
    railway_sd_configs:
      - service: 'ml-service'
        environment: 'production'
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'railway-postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'railway-redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - railway_sd_configs:
        - service: 'alertmanager'
          environment: 'production'

rule_files:
  - "/etc/prometheus/rules/*.yml"
```

#### Grafana Dashboard for Railway
```json
{
  "dashboard": {
    "title": "CRM Inteligente - Railway Production",
    "panels": [
      {
        "title": "Railway Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"crmbet-.*\"}",
            "legendFormat": "{{job}}"
          }
        ]
      },
      {
        "title": "Railway Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "railway_memory_usage_bytes / railway_memory_limit_bytes * 100",
            "legendFormat": "{{service}} Memory %"
          }
        ]
      },
      {
        "title": "Railway CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(railway_cpu_usage_seconds_total[5m]) * 100",
            "legendFormat": "{{service}} CPU %"
          }
        ]
      }
    ]
  }
}
```

## Operational Procedures

### Deployment Checklist

#### Pre-Deployment
- [ ] Code review completed and approved
- [ ] All tests passing (unit, integration, e2e)
- [ ] Security scan completed with no high/critical issues
- [ ] Database migrations tested in staging
- [ ] Performance testing completed
- [ ] Backup verification completed
- [ ] Monitoring alerts configured
- [ ] Rollback plan documented

#### During Deployment
- [ ] Deployment initiated from approved branch
- [ ] Health checks monitoring active
- [ ] Error rates being monitored
- [ ] Performance metrics being tracked
- [ ] Database migration status verified
- [ ] Cache warming completed if needed

#### Post-Deployment
- [ ] Smoke tests passed
- [ ] Critical user journeys verified
- [ ] Performance metrics within SLA
- [ ] Error rates normal
- [ ] Monitoring alerts functional
- [ ] Documentation updated
- [ ] Stakeholders notified

### Incident Response Procedures

#### Severity Levels
- **Critical (P0)**: Complete service outage, data loss, security breach
- **High (P1)**: Major feature unavailable, significant performance degradation
- **Medium (P2)**: Minor feature issues, some users affected
- **Low (P3)**: Cosmetic issues, minimal user impact

#### Response Process
1. **Detection** (< 5 minutes)
   - Automated monitoring alerts
   - User reports
   - Health check failures

2. **Assessment** (< 10 minutes)
   - Determine severity level
   - Activate incident response team
   - Create incident channel

3. **Response** (< 30 minutes for P0/P1)
   - Implement immediate mitigation
   - Communicate status to stakeholders
   - Begin root cause analysis

4. **Recovery**
   - Deploy fix or rollback
   - Verify service restoration
   - Monitor for stability

5. **Post-Incident**
   - Conduct post-mortem
   - Document lessons learned
   - Implement preventive measures

### Backup and Recovery Procedures

#### Automated Backups
```bash
#!/bin/bash
# scripts/backup.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
S3_BUCKET="crmbet-backups"

echo "Starting backup process at $TIMESTAMP"

# Database backup
echo "Backing up PostgreSQL database..."
pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Upload to S3
echo "Uploading to S3..."
aws s3 cp "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz" "s3://$S3_BUCKET/database/"

# Redis backup
echo "Backing up Redis..."
redis-cli --rdb "$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"
aws s3 cp "$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb" "s3://$S3_BUCKET/redis/"

# Application state backup
echo "Backing up application state..."
tar -czf "$BACKUP_DIR/app_state_$TIMESTAMP.tar.gz" /app/logs /app/uploads
aws s3 cp "$BACKUP_DIR/app_state_$TIMESTAMP.tar.gz" "s3://$S3_BUCKET/application/"

# Cleanup old local backups (keep last 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup process completed successfully"
```

#### Recovery Procedures
```bash
#!/bin/bash
# scripts/recovery.sh

set -e

BACKUP_TIMESTAMP="$1"
S3_BUCKET="crmbet-backups"

if [ -z "$BACKUP_TIMESTAMP" ]; then
    echo "Usage: $0 <backup_timestamp>"
    echo "Example: $0 20240115_143000"
    exit 1
fi

echo "Starting recovery process for backup: $BACKUP_TIMESTAMP"

# Download backups from S3
echo "Downloading database backup..."
aws s3 cp "s3://$S3_BUCKET/database/db_backup_$BACKUP_TIMESTAMP.sql.gz" "/tmp/"

echo "Downloading Redis backup..."
aws s3 cp "s3://$S3_BUCKET/redis/redis_backup_$BACKUP_TIMESTAMP.rdb" "/tmp/"

# Stop services
echo "Stopping services..."
railway service stop backend
railway service stop ml-service

# Restore database
echo "Restoring database..."
gunzip -c "/tmp/db_backup_$BACKUP_TIMESTAMP.sql.gz" | psql "$DATABASE_URL"

# Restore Redis
echo "Restoring Redis..."
redis-cli FLUSHALL
redis-cli --rdb "/tmp/redis_backup_$BACKUP_TIMESTAMP.rdb"

# Start services
echo "Starting services..."
railway service start backend
railway service start ml-service

# Verify recovery
echo "Verifying recovery..."
sleep 30
curl -f "https://api.crmbet.com/health"

echo "Recovery process completed successfully"
```

This comprehensive infrastructure deployment guide provides all necessary configurations and procedures for successfully deploying and operating CRM Inteligente in a production Railway environment with enterprise-grade reliability, security, and observability.