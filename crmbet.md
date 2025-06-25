🚀 Prompt Claude Code UltraThink | CRM com Clusterização e Smart Marketing

⸻

🧠 Prompt:

⚙️ Sistema: UltraThink + Hardness + Subtask

🧠 Instrução:
Você é um arquiteto de software sênior especializado em IA, Big Data, CRM e automação. Crie um sistema completo e escalável baseado na seguinte descrição.

⸻

🔥 Descrição do Projeto:

Desenvolver um CRM Inteligente com Machine Learning, que analisa dados do Data Lake e faz clusterização dos usuários com base em comportamento de apostas, preferências de jogos, ticket médio, dias e horários de atividade, além do canal de comunicação preferido. Esse sistema alimenta automaticamente campanhas no Smartico ou qualquer outro CRM.

⸻

🔍 Regras do Sistema:
	1.	🚀 Arquitetura Escalável: Backend robusto, com banco PostgreSQL, Redis, RabbitMQ, deploy em containers Docker (Railway).
	2.	🎲 Machine Learning Pipeline: Clusterização automática dos usuários com KMeans, DBSCAN ou HDBSCAN.
	3.	🔗 Fonte de Dados: Ler dados do Data Lake (ex.: AWS S3) e da tabela tbl_transactions.
	4.	📊 Features para Clusterização:
	•	Jogo favorito / tipo de jogo (ex.: Crash, Cassino, Esportes)
	•	Ticket médio (baixo, médio, alto)
	•	Dias e horários de maior atividade
	•	Canal de comunicação preferido (SMS, WhatsApp, Email)
	5.	🧠 Criação de Clusters: Identificar grupos como:
	•	Gosta de Crash e aposta alto
	•	Joga mais à noite
	•	Prefere cashback
	6.	🔥 DataFrame Final: Gerar tabela df_final_union com os dados do cluster e informações enriquecidas do usuário.
	7.	📤 Integração CRM: Enviar os dados automaticamente para campanhas personalizadas no Smartico via API REST.
	8.	🖥️ Dashboard Operacional:
	•	Visualização de clusters
	•	Análise de comportamento
	•	Criação manual ou automática de campanhas

⸻

🧠 Tarefas em Paralelo (Subtasks):
	1.	🔧 Arquiteto de Infraestrutura:
	•	Define toda a arquitetura: Data Lake, Pipeline de dados, banco SQL, ML pipeline, backend e frontend.
	2.	🏗️ Engenheiro de Dados:
	•	Cria scripts ETL do Data Lake → PostgreSQL.
	•	Faz processamento, limpeza e transformação dos dados.
	3.	🤖 Cientista de Dados:
	•	Implementa os algoritmos de clusterização (KMeans, DBSCAN).
	•	Define as features, testa hiperparâmetros e avalia qualidade dos clusters.
	4.	🖥️ Desenvolvedor Backend:
	•	Cria API REST em Node.js + Express.
	•	Endpoints: /user/:id/segment, /clusters, /campaigns.
	•	Integra com Redis, RabbitMQ e Smartico API.
	5.	🎨 Desenvolvedor Frontend:
	•	Cria um dashboard React + Tailwind com gráficos, lista de clusters, usuários, campanhas e insights.

⸻

🏗️ Entregáveis Gerados:
	•	📜 Documentação Técnica (.md) com:
	•	Arquitetura completa
	•	Diagrama de dados
	•	Descrição dos clusters
	•	🗂️ Estrutura de Pastas organizada:
	•	/backend (Node.js)
	•	/frontend (React)
	•	/ml (Python – clusterização)
	•	/etl (Python ou Node – ingestão de dados)
	•	🔥 Arquivos:
	•	index.js (backend)
	•	App.jsx (frontend)
	•	ml_cluster.py (modelo)
	•	etl_pipeline.py (dados)
	•	🐳 Dockerfile e docker-compose.yaml prontos para deploy.
	•	🔗 Webhook + API Smartico integrados para envio automático dos clusters e perfis.

⸻

🚀 Output Esperado:

🏆 Entregar todo o código backend, frontend, pipelines ML, scripts de integração, além de um arquivo .md documentando toda a arquitetura, explicando o funcionamento do sistema, APIs disponíveis, estrutura dos dados e exemplos de payloads.

⸻

🧠 Finalização do Prompt:

Execute todas essas tarefas de forma paralela, utilizando UltraThink, Subtask e Hardness, com foco em entregar um sistema funcional, escalável e pronto para deploy, com foco na performance, resiliência e facilidade de manutenção.

# 📊 CRM Inteligente com Machine Learning e Clusterização

## 🏗️ Arquitetura Geral
- Data Lake (AWS S3, Google Cloud Storage ou Azure)
- Banco de Dados: PostgreSQL
- Cache e Rate Limit: Redis
- Filas de Processamento: RabbitMQ
- Backend API: Node.js + Express
- Frontend: React + Tailwind
- ML Pipeline: Python (Scikit-Learn + Pandas)
- Deploy: Docker + Railway

## 🔗 Fluxo de Dados
```
Data Lake → ETL → Banco SQL → ML Clusterização → DF Final Union → API → CRM (Smartico) → Campanhas
```

## 🧠 Pipeline de Machine Learning
- Algoritmos: KMeans, DBSCAN ou HDBSCAN
- Features:
  - Tipo de jogo preferido
  - Ticket médio
  - Dias e horários ativos
  - Canal de comunicação preferido
- Saída: clusters segmentando usuários em perfis comportamentais

## 🏗️ Estrutura de Pastas
```
/backend
/frontend
/ml
/etl
/docs
```

## 🚀 Backend API (Node.js)
- Rotas:
  - GET `/user/:id/segment`
  - GET `/clusters`
  - POST `/campaigns`
- Integração com Redis, RabbitMQ e API Smartico

## 🎨 Frontend (React + Tailwind)
- Dashboard:
  - Visualização de clusters
  - Lista de usuários
  - Insights de comportamento
  - Criação de campanhas

## 🤖 Machine Learning (/ml)
- Script: `ml_cluster.py`
- Funções:
  - Processamento de dados
  - Clusterização
  - Geração do dataframe final

## 🔗 Integração com CRM
- API Smartico:
  - Criação e envio de campanhas automáticas
  - Baseado no cluster e comportamento dos usuários

## 🐳 Docker
- Dockerfile
- docker-compose.yaml
- Deploy na Railway

## 📄 Documentação
- Descrição dos clusters
- Exemplo de payloads
- Especificação da API
- Diagrama de arquitetura
