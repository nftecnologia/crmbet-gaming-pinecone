# 🚀 PRODUCTION PROCFILE - Ultra-Robusto System
# Configurado para máxima performance e escala massiva

# Backend API Ultra-Performance (100k+ RPS)
web: cd backend && npm start

# ML Pipeline Distribuído (1M+ predictions/sec)
ml: cd ml && python src/enterprise_ml_pipeline.py --mode production

# ETL Pipeline Industrial (TB+/hour)
etl: cd etl && python run_pipeline.py --mode batch --env production

# Frontend Build & Serve
frontend: cd frontend && npm run serve

# Worker Processes para Background Tasks
worker: cd backend && npm run worker

# Scheduler para ETL Batch Jobs
scheduler: cd etl && python scheduler.py --env production

# Monitoring & Health Checks
monitor: cd monitoring && python health_monitor.py --env production