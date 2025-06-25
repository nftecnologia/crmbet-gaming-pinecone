# CRM ML Dashboard - Frontend

Dashboard operacional para CRM ML desenvolvido em React com foco em UX excepcional e visualizações de dados avançadas.

## ⚡ Tecnologias

- **React 18** - Interface moderna e reativa
- **Vite** - Build tool ultra-rápido
- **Tailwind CSS** - Framework CSS utilitário
- **Chart.js** - Visualizações de dados interativas
- **Framer Motion** - Animações fluidas
- **Zustand** - Gerenciamento de estado
- **React Hook Form** - Formulários performáticos
- **Axios** - Cliente HTTP

## 🚀 Começando

### Pré-requisitos

- Node.js 18+
- npm ou yarn

### Instalação

1. **Clone o repositório**
   ```bash
   git clone <repository-url>
   cd crmbet/frontend
   ```

2. **Instale as dependências**
   ```bash
   npm install
   ```

3. **Configure o ambiente**
   ```bash
   cp .env.example .env
   # Edite o arquivo .env com suas configurações
   ```

4. **Execute o servidor de desenvolvimento**
   ```bash
   npm run dev
   ```

5. **Acesse o dashboard**
   - Abra [http://localhost:3000](http://localhost:3000)

## 📋 Scripts Disponíveis

- `npm run dev` - Servidor de desenvolvimento
- `npm run build` - Build para produção
- `npm run preview` - Preview do build
- `npm run lint` - Verificação de código
- `npm run lint:fix` - Correção automática
- `npm run format` - Formatação de código

## 🏗️ Estrutura do Projeto

```
src/
├── components/          # Componentes reutilizáveis
│   ├── Layout.jsx      # Container principal
│   ├── Sidebar.jsx     # Navegação lateral
│   ├── Header.jsx      # Cabeçalho
│   ├── ClusterCard.jsx # Card de cluster
│   ├── UserTable.jsx   # Tabela de usuários
│   └── CampaignForm.jsx # Formulário de campanha
├── pages/              # Páginas principais
│   ├── Dashboard.jsx   # Página inicial
│   ├── Clusters.jsx    # Gerenciamento de clusters
│   ├── Users.jsx       # Lista de usuários
│   ├── Campaigns.jsx   # Campanhas de marketing
│   └── Analytics.jsx   # Insights e métricas
├── hooks/              # Hooks customizados
│   ├── useCluster.js   # Lógica de clusters
│   ├── useCampaign.js  # Lógica de campanhas
│   └── useUser.js      # Lógica de usuários
├── services/           # Integração com APIs
│   └── api.js          # Cliente HTTP configurado
├── stores/             # Estado global (Zustand)
│   └── useStore.js     # Stores da aplicação
├── styles/             # Estilos globais
│   └── index.css       # CSS com Tailwind
├── utils/              # Funções utilitárias
│   └── index.js        # Helpers diversos
└── App.jsx             # Componente raiz
```

## 🎨 Design System

### Cores Principais

- **Primary**: Azul (#3b82f6)
- **Success**: Verde (#22c55e)
- **Warning**: Amarelo (#f59e0b)
- **Danger**: Vermelho (#ef4444)
- **Secondary**: Cinza (#64748b)

### Componentes de Design

- **Buttons**: `.btn-primary`, `.btn-secondary`, `.btn-ghost`
- **Cards**: `.card`, `.card-header`, `.card-body`, `.card-footer`
- **Forms**: `.form-input`, `.form-label`, `.form-error`
- **Badges**: `.badge-success`, `.badge-warning`, `.badge-danger`
- **Tables**: `.data-table` com estilização responsiva

## 📊 Funcionalidades

### Dashboard Principal
- Métricas em tempo real
- Gráficos interativos
- Atividade recente
- Insights de ML

### Gerenciamento de Clusters
- Visualização de segmentos
- Métricas por cluster
- Edição e criação
- Distribuição visual

### Campanhas de Marketing
- Criação com formulário multi-step
- Segmentação por cluster
- Métricas de performance
- Controles de execução

### Analytics Avançado
- Funil de conversão
- Heatmap de atividade
- ROI por campanha
- Recomendações de ML

### Gerenciamento de Usuários
- Tabela com paginação
- Filtros avançados
- Exportação de dados
- Visualização por cluster

## 🔧 APIs Integradas

### Endpoints Principais

- **Users**: `/api/users` - CRUD de usuários
- **Clusters**: `/api/clusters` - Segmentos ML
- **Campaigns**: `/api/campaigns` - Campanhas
- **Analytics**: `/api/analytics` - Métricas
- **ML**: `/api/ml` - Modelos e predições

### Autenticação

A aplicação usa JWT tokens armazenados no localStorage com refresh automático.

## 📱 Responsividade

- **Mobile First**: Design otimizado para móveis
- **Breakpoints**: sm (640px), md (768px), lg (1024px), xl (1280px)
- **Sidebar**: Colapsível em telas pequenas
- **Tabelas**: Scroll horizontal em móveis
- **Cards**: Stack vertical em dispositivos menores

## 🎯 Performance

### Otimizações Implementadas

- **Code Splitting**: Carregamento sob demanda
- **Lazy Loading**: Componentes assíncronos
- **Memoização**: React.memo e useMemo
- **Debounce**: Buscas com delay
- **Virtual Scrolling**: Listas grandes

### Métricas Alvo

- **FCP**: < 1.5s
- **LCP**: < 2.5s
- **CLS**: < 0.1
- **Bundle Size**: < 500KB gzipped

## 🧪 Testes

### Executar Testes

```bash
npm test                # Testes unitários
npm run test:coverage   # Cobertura de código
npm run test:e2e        # Testes end-to-end
```

### Estratégia de Testes

- **Unit**: Componentes isolados
- **Integration**: Fluxos de usuário
- **E2E**: Jornadas completas
- **Visual**: Regression testing

## 🚀 Deploy

### Build de Produção

```bash
npm run build
```

### Variáveis de Ambiente

```bash
VITE_API_URL=https://api.production.com
VITE_APP_ENV=production
VITE_ANALYTICS_ID=GA_MEASUREMENT_ID
```

### Plataformas Suportadas

- **Vercel**: Deploy automático
- **Netlify**: Build otimizado
- **AWS S3**: Hospedagem estática
- **Docker**: Container pronto

## 🔍 Monitoramento

### Analytics

- **Google Analytics**: Comportamento do usuário
- **Sentry**: Monitoramento de erros
- **Web Vitals**: Métricas de performance

### Logs

- **Console**: Desenvolvimento
- **API Calls**: Request/Response timing
- **User Actions**: Eventos importantes

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

### Convenções

- **Commits**: Conventional Commits
- **Branches**: feature/, bugfix/, hotfix/
- **Código**: ESLint + Prettier
- **Testes**: Cobertura mínima de 80%

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](../LICENSE) para detalhes.

## 🆘 Suporte

- **Documentação**: [Wiki do projeto]
- **Issues**: [GitHub Issues]
- **Discord**: [Canal da equipe]
- **Email**: dev@crmml.com

---

**Desenvolvido com ❤️ pela equipe CRM ML**